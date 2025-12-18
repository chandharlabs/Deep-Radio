import socket
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("192.168.0.106", 2152))

print("Listening for GTP-U packets...")
data, addr = sock.recvfrom(4096)
sock.close()

gtp_header = data[:8]
gtp_flags, gtp_type, gtp_len = struct.unpack("!BBH", gtp_header[:4])
gtp_teid = struct.unpack("!I", gtp_header[4:8])[0]
qfi = data[8]
message = data[9:]

print(f"From: {addr}")
print(f"GTP Flags: {gtp_flags:#04x}, Type: {gtp_type:#04x}, Length: {gtp_len}")
print(f"TEID: {gtp_teid:#010x}")
print(f"QFI: {qfi}")
print(f"GTP payload: {message.decode(errors='ignore')}")
