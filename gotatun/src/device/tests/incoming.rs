// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// This file incorporates work covered by the following copyright and
// permission notice:
//
//   Copyright (c) Mullvad VPN AB. All rights reserved.
//
// SPDX-License-Identifier: MPL-2.0

//! Tests that drive [`DeviceState::handle_incoming`] directly with controllable transports.

use std::{
    io,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use ipnetwork::Ipv4Network;
use tokio::{sync::mpsc, task::JoinHandle, time::timeout};
use x25519_dalek::{PublicKey, StaticSecret};
use zerocopy::IntoBytes;

use super::mock;
use crate::{
    device::{
        DeviceBuilder, DeviceState, HANDSHAKE_RATE_LIMIT, Peer, transports::DeviceTransports,
    },
    noise::{TimerParams, Tunn, TunnResult, index_table::IndexTable, rate_limiter::RateLimiter},
    packet::{Ip, Packet, PacketBufPool, WgHandshakeResp, WgKind},
    tun::IpSend,
    udp::{UdpRecv, UdpSend, socket::UdpSocketFactory},
};

const WAIT: Duration = Duration::from_secs(2);

type SuspendedDevice = crate::device::Device<(UdpSocketFactory, mock::MockTun, mock::MockTun)>;

type Datagram = (Packet, SocketAddr);

/// [`UdpRecv`] that yields datagrams pushed into a channel.
struct ChannelUdpRecv(mpsc::Receiver<Datagram>);

impl UdpRecv for ChannelUdpRecv {
    type RecvManyBuf = ();

    async fn recv_from(&mut self, _pool: &mut PacketBufPool) -> io::Result<Datagram> {
        self.0
            .recv()
            .await
            .ok_or_else(|| io::ErrorKind::UnexpectedEof.into())
    }
}

/// [`UdpSend`] that records every datagram and optionally fails every send.
#[derive(Clone)]
struct RecordingUdpSend {
    sent: mpsc::UnboundedSender<Datagram>,
    fail: bool,
}

impl UdpSend for RecordingUdpSend {
    type SendManyBuf = ();

    async fn send_to(&self, packet: Packet, destination: SocketAddr) -> io::Result<()> {
        let _ = self.sent.send((packet, destination));
        if self.fail {
            Err(io::Error::other("send failed"))
        } else {
            Ok(())
        }
    }
}

fn recording_udp(fail: bool) -> (RecordingUdpSend, mpsc::UnboundedReceiver<Datagram>) {
    let (sent, rx) = mpsc::unbounded_channel();
    (RecordingUdpSend { sent, fail }, rx)
}

/// [`IpSend`] that never completes a send, like a TUN device that stopped draining.
struct StalledIpSend(mpsc::UnboundedSender<Packet<Ip>>);

impl IpSend for StalledIpSend {
    async fn send(&mut self, packet: Packet<Ip>) -> io::Result<()> {
        let _ = self.0.send(packet);
        std::future::pending().await
    }
}

/// A device whose own connection is not running, so tests can drive its tasks directly.
async fn suspended_device(private_key: StaticSecret, peer: Peer) -> SuspendedDevice {
    let (tun, _app_tx, _app_rx) = mock::mock_tun();
    DeviceBuilder::new()
        .with_private_key(private_key)
        .with_ip(tun)
        .with_default_udp()
        .with_peer(peer)
        .suspended(true)
        .build()
        .await
        .expect("create device")
}

fn tunn(private_key: StaticSecret, peer_public_key: PublicKey) -> Tunn {
    let rate_limiter = Arc::new(RateLimiter::new(
        &PublicKey::from(&private_key),
        HANDSHAKE_RATE_LIMIT,
    ));
    Tunn::new(
        private_key,
        peer_public_key,
        None,
        None,
        IndexTable::from_os_rng(),
        rate_limiter,
    )
}

fn spawn_incoming<T: DeviceTransports>(
    device: &crate::device::Device<T>,
    tun_tx: impl IpSend,
    udp_tx: impl UdpSend + 'static,
) -> (
    JoinHandle<Result<(), crate::device::Error>>,
    mpsc::Sender<Datagram>,
) {
    let (udp_in, udp_rx) = mpsc::channel(16);
    let task = tokio::spawn(DeviceState::handle_incoming(
        Arc::downgrade(&device.inner),
        tun_tx,
        udp_tx,
        ChannelUdpRecv(udp_rx),
        PacketBufPool::new(16),
    ));
    (task, udp_in)
}

async fn next_sent(sent: &mut mpsc::UnboundedReceiver<Datagram>) -> WgKind {
    let (packet, _) = timeout(WAIT, sent.recv())
        .await
        .expect("timed out waiting for a sent datagram")
        .expect("sender dropped");
    packet.try_into_wg().expect("sent a WireGuard packet")
}

fn decrypt(tunn: &mut Tunn, packet: WgKind) -> Packet {
    match tunn.handle_incoming_packet(packet) {
        TunnResult::WriteToTunnel(packet) => packet,
        other => panic!("expected a decrypted packet, got {other:?}"),
    }
}

async fn send_handshake_init(
    alice: &mut Tunn,
    udp_in: &mpsc::Sender<Datagram>,
    udp_sent: &mut mpsc::UnboundedReceiver<Datagram>,
    alice_addr: SocketAddr,
) -> Packet<WgHandshakeResp> {
    let init = alice.format_handshake_initiation(false).unwrap();
    udp_in
        .send((WgKind::from(init).into(), alice_addr))
        .await
        .unwrap();
    match next_sent(udp_sent).await {
        WgKind::HandshakeResp(resp) => resp,
        other => panic!("expected HandshakeResp, got {other:?}"),
    }
}

async fn confirm_responder_handshake(
    alice: &mut Tunn,
    resp: Packet<WgHandshakeResp>,
    udp_in: &mpsc::Sender<Datagram>,
    alice_addr: SocketAddr,
) {
    let TunnResult::WriteToNetwork(keepalive @ WgKind::Data(_)) =
        alice.handle_incoming_packet(WgKind::HandshakeResp(resp))
    else {
        panic!("expected alice to complete the handshake with a keepalive");
    };
    udp_in.send((keepalive.into(), alice_addr)).await.unwrap();
}

/// A backpressured TUN send in `handle_incoming` must not hold the device or peer locks, which
/// would stall `handle_timers`, `handle_outgoing`, and the configuration API.
#[tokio::test]
#[test_log::test]
async fn blocked_tun_send_does_not_hold_locks() {
    let alice_key = StaticSecret::random();
    let bob_key = StaticSecret::random();
    let alice_addr: SocketAddr = (Ipv4Addr::new(10, 0, 0, 1), 51820).into();

    let mut alice = tunn(alice_key.clone(), PublicKey::from(&bob_key));
    let keepalive_timeout = Duration::from_millis(100);
    let alice_peer = Peer::new(PublicKey::from(&alice_key))
        .with_endpoint(alice_addr)
        .with_allowed_ip(Ipv4Network::new(Ipv4Addr::UNSPECIFIED, 0).unwrap().into())
        .dangerously_with_timer_params(TimerParams {
            keepalive_timeout: keepalive_timeout..=keepalive_timeout,
            ..TimerParams::default()
        });
    let bob = suspended_device(bob_key, alice_peer).await;

    let (tun_tx, mut delivered) = mpsc::unbounded_channel();
    let (udp_tx, mut udp_sent) = recording_udp(false);
    let (incoming, udp_in) = spawn_incoming(&bob, StalledIpSend(tun_tx), udp_tx);

    // Complete a handshake through `handle_incoming`.
    let response = send_handshake_init(&mut alice, &udp_in, &mut udp_sent, alice_addr).await;
    confirm_responder_handshake(&mut alice, response, &udp_in, alice_addr).await;

    // Deliver a data packet that `handle_incoming` cannot hand to the stalled TUN device.
    let blocked = mock::packet(b"blocked");
    let data = alice
        .handle_outgoing_packet(blocked.clone().into_bytes(), None)
        .expect("alice has a session");
    udp_in.send((data.into(), alice_addr)).await.unwrap();
    let stalled = timeout(WAIT, delivered.recv())
        .await
        .expect("handle_incoming should reach the TUN send")
        .unwrap();
    assert_eq!(stalled.as_bytes(), blocked.as_bytes());
    assert!(!incoming.is_finished(), "handle_incoming must stay blocked");

    // Peer stats take the device read lock and the peer lock.
    let peers = timeout(WAIT, bob.peers())
        .await
        .expect("peers() must not wait for the blocked TUN send");
    assert!(peers[0].stats.last_handshake.is_some());

    // The data packet armed bob's passive keepalive, which `handle_timers` sends.
    #[cfg(feature = "mock_instant")]
    mock_instant::thread_local::MockClock::advance(Duration::from_secs(1));
    let (timers_udp, mut timers_sent) = recording_udp(false);
    let timers = tokio::spawn(DeviceState::handle_timers(
        Arc::downgrade(&bob.inner),
        timers_udp,
    ));
    let keepalive = decrypt(&mut alice, next_sent(&mut timers_sent).await);
    assert!(
        keepalive.is_empty(),
        "handle_timers should send a keepalive"
    );

    // `handle_outgoing` encrypts and sends a packet read from a TUN device.
    let (outgoing_udp, mut outgoing_sent) = recording_udp(false);
    let (tun, app_tx, _app_rx) = mock::mock_tun();
    let outgoing = tokio::spawn(DeviceState::handle_outgoing(
        Arc::downgrade(&bob.inner),
        tun,
        outgoing_udp,
        PacketBufPool::new(16),
    ));
    let reply = mock::packet(b"reply");
    app_tx.send(reply.clone()).await;
    let received = decrypt(&mut alice, next_sent(&mut outgoing_sent).await);
    assert_eq!(
        &received.as_bytes()[..reply.as_bytes().len()],
        reply.as_bytes()
    );

    // Reconfiguration takes the device write lock.
    timeout(WAIT, bob.write(async |_| ()))
        .await
        .expect("write() must not wait for the blocked TUN send")
        .unwrap();

    incoming.abort();
    timers.abort();
    outgoing.abort();
}

/// A failure to send a cookie reply must not stop `handle_incoming`, which would take down the
/// whole device.
#[tokio::test]
#[test_log::test]
async fn cookie_reply_send_error_does_not_stop_handle_incoming() {
    let alice_key = StaticSecret::random();
    let bob_key = StaticSecret::random();
    let loaded_addr: SocketAddr = (Ipv4Addr::new(10, 0, 0, 1), 51820).into();
    let other_addr: SocketAddr = (Ipv4Addr::new(10, 0, 0, 2), 51820).into();

    let mut alice = tunn(alice_key.clone(), PublicKey::from(&bob_key));
    let alice_peer = Peer::new(PublicKey::from(&alice_key))
        .with_allowed_ip(Ipv4Network::new(Ipv4Addr::UNSPECIFIED, 0).unwrap().into());
    let bob = suspended_device(bob_key, alice_peer).await;

    let (tun_tx, _delivered) = mpsc::unbounded_channel();
    let (udp_tx, mut udp_sent) = recording_udp(true);
    let (incoming, udp_in) = spawn_incoming(&bob, StalledIpSend(tun_tx), udp_tx);

    // Exhaust the handshake budget of one source address so bob answers it with cookies.
    let init = alice.format_handshake_initiation(false).unwrap();
    let rate_limiter = bob.inner.read().await.rate_limiter.clone().unwrap();
    for _ in 0..HANDSHAKE_RATE_LIMIT {
        rate_limiter
            .verify_handshake(loaded_addr, init.clone())
            .expect("under the rate limit");
    }

    udp_in
        .send((WgKind::from(init.clone()).into(), loaded_addr))
        .await
        .unwrap();
    assert!(
        matches!(next_sent(&mut udp_sent).await, WgKind::CookieReply(_)),
        "bob should attempt to send a cookie reply"
    );

    // The same initiation from an address under the limit is still answered.
    udp_in
        .send((WgKind::from(init).into(), other_addr))
        .await
        .unwrap();
    assert!(
        matches!(next_sent(&mut udp_sent).await, WgKind::HandshakeResp(_)),
        "handle_incoming should keep processing packets after a failed cookie reply"
    );
    assert!(!incoming.is_finished());

    incoming.abort();
}

/// Confirming a responder session with an empty keepalive must drain staged outbound packets
/// rather than stranding them in `packet_queue`.
#[tokio::test]
#[test_log::test]
async fn responder_confirmation_keepalive_drains_queued_outbound_packets() {
    let alice_key = StaticSecret::random();
    let bob_key = StaticSecret::random();
    let alice_addr: SocketAddr = (Ipv4Addr::new(10, 0, 0, 1), 51820).into();

    let mut alice = tunn(alice_key.clone(), PublicKey::from(&bob_key));
    let alice_peer = Peer::new(PublicKey::from(&alice_key))
        .with_endpoint(alice_addr)
        .with_allowed_ip(Ipv4Network::new(Ipv4Addr::UNSPECIFIED, 0).unwrap().into());
    let bob = suspended_device(bob_key, alice_peer).await;

    let (tun_tx, _delivered) = mpsc::unbounded_channel();
    let (udp_tx, mut udp_sent) = recording_udp(false);
    let (incoming, udp_in) = spawn_incoming(&bob, StalledIpSend(tun_tx), udp_tx.clone());

    // Alice initiates and Bob answers, creating an unconfirmed responder session on Bob.
    let resp = send_handshake_init(&mut alice, &udp_in, &mut udp_sent, alice_addr).await;

    // Bob's app tries to send an outbound packet before Alice's confirmation arrives.
    // The packet is queued and Bob starts a local initiation (which we drop).
    let (outgoing_udp, mut outgoing_sent) = recording_udp(false);
    let (tun, app_tx, _app_rx) = mock::mock_tun();
    let outgoing = tokio::spawn(DeviceState::handle_outgoing(
        Arc::downgrade(&bob.inner),
        tun,
        outgoing_udp,
        PacketBufPool::new(16),
    ));
    let payload = mock::packet(b"first-staged");
    app_tx.send(payload.clone()).await;
    assert!(
        matches!(
            next_sent(&mut outgoing_sent).await,
            WgKind::HandshakeInit(_)
        ),
        "bob should queue the packet and initiate"
    );

    // Alice completes the handshake and sends her confirming empty keepalive to Bob.
    confirm_responder_handshake(&mut alice, resp, &udp_in, alice_addr).await;

    // Bob must drain the staged outbound packet to UDP on responder confirmation.
    let drained = next_sent(&mut udp_sent).await;
    let decrypted = decrypt(&mut alice, drained);
    assert_eq!(
        &decrypted.as_bytes()[..payload.as_bytes().len()],
        payload.as_bytes()
    );

    incoming.abort();
    outgoing.abort();
}

/// `Device::force_handshake` must complete when our own initiation is lost and the peer
/// completes a responder-side handshake with us.
#[tokio::test]
#[test_log::test]
async fn force_handshake_completes_when_responder_session_is_confirmed() {
    let alice_key = StaticSecret::random();
    let bob_key = StaticSecret::random();
    let alice_pub = PublicKey::from(&alice_key);
    let alice_addr: SocketAddr = (Ipv4Addr::new(10, 0, 0, 1), 51820).into();

    let mut alice = tunn(alice_key.clone(), PublicKey::from(&bob_key));
    let alice_peer = Peer::new(alice_pub)
        .with_endpoint(alice_addr)
        .with_allowed_ip(Ipv4Network::new(Ipv4Addr::UNSPECIFIED, 0).unwrap().into());
    let bob = suspended_device(bob_key, alice_peer).await;

    let (tun_tx, _delivered) = mpsc::unbounded_channel();
    let (udp_tx, mut udp_sent) = recording_udp(false);
    let (incoming, udp_in) = spawn_incoming(&bob, StalledIpSend(tun_tx), udp_tx);

    // Subscribe via PeerState::force_handshake (sending initiation, which is lost).
    let mut completed = {
        let device = bob.inner.read().await;
        let peer_arc = device.peers.get(&alice_pub).unwrap();
        let mut peer = peer_arc.lock().await;
        peer.force_handshake()
            .unwrap()
            .expect("no active session yet")
            .completed
    };

    // Alice initiates and confirms a responder session on Bob instead.
    let resp = send_handshake_init(&mut alice, &udp_in, &mut udp_sent, alice_addr).await;
    confirm_responder_handshake(&mut alice, resp, &udp_in, alice_addr).await;

    timeout(WAIT, completed.changed())
        .await
        .expect("responder promotion must wake force_handshake waiters")
        .expect("peer must stay alive");

    incoming.abort();
}
