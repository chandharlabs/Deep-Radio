import socket
import struct
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from commpy.channelcoding import Trellis, conv_encode

# GTP-U Layer
def receive_gtp_data(local_ip="192.168.0.109", local_port=2152):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((local_ip, local_port))
    print(f"[GTP] Listening on {local_ip}:{local_port}")
    data, addr = sock.recvfrom(4096)
    print(f"[GTP] Received {len(data)} bytes from {addr}")
    #print(f"[GTP] Raw data (hex): {data.hex()}")
    return data

# SDAP Layer
def sdap_process(gtp_payload):
    qfi = gtp_payload[0]
    payload = gtp_payload[1:]
    #print(f"\n[SDAP] Processing:")
    print(f" - QFI: {qfi}")
    print(f" - Payload length: {len(payload)} bytes")
    #print(f" - Payload (hex): {payload.hex()}")
    #print(f" - Payload (ASCII): {payload.decode('ascii', errors='replace')}")
    return payload

# PDCP Layer
def pdcp_layer(data, key=b'secretkey1234567'):
    sn = struct.pack("!H", 1)  # Sequence number
    iv = get_random_bytes(8)    # Initialization Vector
    #print(f"\n[PDCP] Encrypting:")
    #print(f" - SN: {sn.hex()}")
    #print(f" - IV: {iv.hex()}")
    #print(f" - Plaintext (hex): {data.hex()}")
    
    cipher = AES.new(key, AES.MODE_CTR, nonce=iv[:4])
    encrypted = cipher.encrypt(data)
    result = sn + iv + encrypted
    
    #print(f" - Ciphertext (hex): {result.hex()}")
    #print(f" - Total PDCP PDU: {len(result)} bytes")
    return result

# RLC Layer
def rlc_segment(data, sn_start=0, max_pdu_size=30):
    segments = []
    sn = sn_start
    print(f"\n[RLC] Segmenting {len(data)} bytes into PDUs (max {max_pdu_size} bytes each):")
    
    for i in range(0, len(data), max_pdu_size):
        chunk = data[i:i + max_pdu_size]
        pdu = struct.pack("!H", sn) + chunk
        segments.append(pdu)
        print(f" - PDU {sn}: {len(chunk)} bytes payload (hex: {chunk.hex()})")
        sn = (sn + 1) % 65536
    
    print(f"Total segments: {len(segments)}")
    return segments

# CRC Calculation
def crc16(data: bytes):
    poly = 0x11021
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc

# MAC Layer
def mac_add_crc(rlc_pdu):
    crc_val = crc16(rlc_pdu)
    #print(f"\n[MAC] CRC Calculation:")
    #print(f" - Input data (hex): {rlc_pdu.hex()}")
    #print(f" - Calculated CRC: {hex(crc_val)}")
    return rlc_pdu + struct.pack("!H", crc_val)

# PHY Layer
def convolutional_encode(bits):
    # Ensure proper padding for byte conversion
    pad_len = (8 - (len(bits) % 8)) % 8
    padded = np.pad(bits, (0, pad_len), 'constant')
    
    trellis = Trellis(np.array([7]), np.array([[0o133, 0o171]]))
    encoded = conv_encode(padded, trellis, 'term')
    np.set_printoptions(threshold=np.inf)
    
    #print(f"\n[PHY] Convolutional Encoding:")
    #print(f" - Input bits: {len(bits)} (+{pad_len} padding)")
    #print(f" - Encoded bits: {len(encoded)}")
    #print(f" - Encoded bits: {encoded}")
    #print(f"\n - Actual bits: {padded}")
    return encoded

# Reliable UDP sender
def reliable_udp_send(ip, port, data, max_retries=3):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    success = False
    
    for attempt in range(max_retries):
        try:
            sock.sendto(data, (ip, port))
            print(f"\n[UDP] Transmission attempt {attempt+1}:")
            print(f" - Sent {len(data)} bytes to {ip}:{port}")
            print(f" - bytes : {data.hex()}...")
            success = True
            break
        except Exception as e:
            print(f"[UDP] Error (attempt {attempt+1}): {e}")
    
    sock.close()
    return success

# Main full-stack TX
def pc2_full_stack_tx():
    print("=== 5G-LITE TRANSMITTER (Press Ctrl+C to stop) ===")
    
    try:
        while True:
            # 1. GTP Layer
            gtp_packet = receive_gtp_data()
            if len(gtp_packet) < 8:
                print("[ERROR] Invalid GTP packet - too short")
                continue
            
            # 2. SDAP Layer
            sdap_pdu = sdap_process(gtp_packet[8:])  # Skip GTP header
            
            # 3. PDCP Layer
            pdcp_sdu = pdcp_layer(sdap_pdu)
            
            # 4. RLC Layer
            rlc_pdus = rlc_segment(pdcp_sdu)
            
            # Process each RLC PDU
            for rlc_pdu in rlc_pdus:
                # 5. MAC Layer
                mac_pdu = mac_add_crc(rlc_pdu)
                
                # 6. PHY Layer
                bits = np.unpackbits(np.frombuffer(mac_pdu, dtype=np.uint8))
                fec_bits = convolutional_encode(bits)
                fec_bytes = np.packbits(fec_bits).tobytes()
                
                # 7. Transmission
                if not reliable_udp_send("192.168.0.116", 5005, fec_bytes):
                    print("[ERROR] Failed to transmit after max retries")
                    break
            
            print("\n--- Packet Transmission Complete ---\n")
    
    except KeyboardInterrupt:
        print("\n[INFO] Transmission terminated by user (Ctrl+C).")

# Entrypoint
if __name__ == "__main__":
    pc2_full_stack_tx()

