import numpy as np
from Crypto.Cipher import AES
from commpy.channelcoding import Trellis, viterbi_decode
import struct
import socket

'''def receive_fec_data(local_ip="192.168.0.116", local_port=5005):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((local_ip, local_port))
    print("\n[PHY] Waiting for UDP packet...")
    data, addr = sock.recvfrom(4096)
    #print(f"[PHY] Received {len(data)} bytes from {addr}")
    #print(f"[PHY] Raw data (hex): {data.hex()}")
    return data'''
    


def receive_fec_data(local_ip="192.168.0.116", local_port=5005, timeout_after_first=5.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((local_ip, local_port))
    sock.settimeout(None)  # Wait forever for the first packet

    print("[Receiver] Waiting for first packet...")
    try:
        data, addr = sock.recvfrom(4096)
        print(f"[Receiver] Received first packet from {addr}")
        packets = [data]

        # Now set timeout for next packets
        sock.settimeout(timeout_after_first)
        while True:
            try:
                data, addr = sock.recvfrom(4096)
                print(f"[Receiver] Received additional packet from {addr}")
                packets.append(data)
            except socket.timeout:
                print(f"[Receiver] No more packets for {timeout_after_first} seconds. Exiting.")
                break
    finally:
        sock.close()

    return packets

    

def bits_to_bytes(bits):
    pad_len = (8 - (len(bits) % 8)) % 8
    padded = np.pad(bits, (0, pad_len), 'constant')
    bytes_data = np.packbits(padded).tobytes()
    #print(f"[PHY] Bit to byte conversion:")
    #print(f" - Input bits: {len(bits)}")
    #print(f" - Padded bits: {len(padded)}")
    #print(f" - Output bytes: {len(bytes_data)}")
    return bytes_data

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

def mac_strip_crc(data_bytes):
    #print("\n[MAC] CRC Verification:")
    #print(f" - Received data (hex): {data_bytes.hex()}")
    
    if len(data_bytes) < 2:
        print("[MAC] Packet too short for CRC")
        return None
    
    payload = data_bytes[:-2]
    received_crc = struct.unpack("!H", data_bytes[-2:])[0]
    calculated_crc = crc16(payload)
    
    #print(f" - Payload (hex): {payload.hex()}")
    #print(f" - Received CRC: {hex(received_crc)}")
   # print(f" - Calculated CRC: {hex(calculated_crc)}")
    
    if received_crc != calculated_crc:
        print("[MAC] CRC mismatch!")
        return None
    
    print("[MAC] CRC verified successfully")
    return payload

def convolutional_decode(bits):
    """Robust convolutional decoding"""
    if len(bits) % 2 != 0:
        print("[PHY] Error: Odd number of bits for rate 1/2 code")
        return None
    
    trellis = Trellis(np.array([7]), np.array([[0o133, 0o171]]))
    symbols = np.array(bits, dtype=float) * 2 - 1  # Convert to ±1
    
    try:
        decoded = viterbi_decode(bits, trellis, tb_depth=5*trellis.total_memory, decoding_type='hard')
        # Remove termination bits
        decoded = decoded[:-trellis.total_memory]
        #print(f"[PHY] Decoded bits -> {decoded} bits")
        #print(f"[PHY] Decoded {len(bits)} bits -> {len(decoded)} bits")
        return decoded
    except Exception as e:
        print(f"[PHY] Decoding failed: {e}")
        return None


def rlc_reassemble(data):
    print("\n[RLC] Processing:")
    print(f" - Received data (hex): {data.hex()}")
    
    if len(data) < 2:
        print("[RLC] Packet too short for sequence number")
        return None
    
    sn = struct.unpack("!H", data[:2])[0]
    payload = data[2:]
    
    print(f" - Sequence number: {sn}")
    print(f" - Payload length: {len(payload)} bytes")
    print(f" - Payload (hex): {payload.hex()}")
    
    return payload

def pdcp_decrypt(payload, key=b'secretkey1234567'):
    #print("\n[PDCP] Decrypting:")
    #print(f" - Input data (hex): {payload.hex()}")
    
    if len(payload) < 10:
        print("[PDCP] Packet too short for header")
        return None
    
    sn = payload[:2]
    iv = payload[2:10]
    ciphertext = payload[10:]
    
    #print(f" - SN: {sn.hex()}")
    #print(f" - IV: {iv.hex()}")
    #print(f" - Ciphertext (hex): {ciphertext.hex()}")
    
    cipher = AES.new(key, AES.MODE_CTR, nonce=iv[:4])
    decrypted = cipher.decrypt(ciphertext)
    
    #print(f" - Decrypted (hex): {decrypted.hex()}")
    return decrypted

def sdap_reconstruct(payload, qfi=0x01):
    #print("\n[SDAP] Reconstructing:")
    #print(f" - Input payload (hex): {payload.hex()}")
    reconstructed = bytes([qfi]) + payload
    #print(f" - Reconstructed packet (hex): {reconstructed.hex()}")
    return reconstructed

def pc2_full_stack_rx():
    print("\n=== Starting Receiver ===")
    fec_data = receive_fec_data()
    print(fec_data)
    
    # Convert to bits
    fec_bits = np.unpackbits(np.frombuffer(fec_data, dtype=np.uint8))[:-2]
    #print(fec_bits)
    # Convolutional decode
    try:
        decoded_bits = convolutional_decode(fec_bits)
     #   print(f"[PHY] Decoded bits -> {decoded_bits}")
    except Exception as e:
        print(f"[PHY] Decoding failed: {e}")
        return

    # Convert to bytes
    mac_bytes = bits_to_bytes(decoded_bits)
    
    # Verify CRC
    payload = mac_strip_crc(mac_bytes)
    if payload is None:
        return
    
    # RLC processing
    rlc_data = rlc_reassemble(payload)
    if rlc_data is None:
        return
    
    # PDCP decryption
    pdcp_data = pdcp_decrypt(rlc_data)
    if pdcp_data is None:
        return
    
    # SDAP reconstruction
    final_payload = sdap_reconstruct(pdcp_data)
    
    print("\n[RX] Final Payload:")
    print(f"Length: {len(final_payload)} bytes")
    print(f"Hex: {final_payload.hex()}")
    try:
        print(f"ASCII: {final_payload.decode('ascii')}")
    except UnicodeDecodeError:
        print("ASCII: [contains non-ASCII characters]")

if __name__ == "__main__":
    while True:
        pc2_full_stack_rx()
