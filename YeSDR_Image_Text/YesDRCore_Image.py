import socket
import struct
from PIL import Image
import io
import base64

def send_gtp_message(message_text, dest_ip="192.168.0.109", dest_port=2152):
    """Send a GTP-U formatted message with QFI header"""
    qfi = 9  # Quality of Service Flow Identifier
    message = bytes([qfi]) + message_text.encode()
    
    # GTP Header (version=1, PT=1, no extension)
    gtp_flags = 0x30
    gtp_msg_type = 0xFF  # G-PDU
    gtp_teid = 0xABCDEF01
    gtp_length = len(message)
    
    # Pack header
    gtp_header = struct.pack("!BBH", gtp_flags, gtp_msg_type, gtp_length)
    gtp_teid_bytes = struct.pack("!I", gtp_teid)
    gtp_packet = gtp_header + gtp_teid_bytes + message
    
    # Send via UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(gtp_packet, (dest_ip, dest_port))
    sock.close()
    
    print(f"Sent: {message_text} (QFI={qfi})")

def interactive_mode():
    print("GTP-U Transmitter (Interactive Mode)")
    print("Enter messages to send (press Enter alone to exit):")
    
    while True:
        try:
            user_input = input("> ")
            if not user_input:
                break
                
            image_path = "/home/shiva/Downloads/sample.jpg"  # Replace with your image path
            image = Image.open(image_path)

# Save image to bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")  # or PNG
            binary_data = buffer.getvalue()

# Encode binary data to base64 string
            base64_str = base64.b64encode(binary_data).decode('utf-8')

            send_gtp_message(base64_str)            
            #send_gtp_message(user_input)
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    interactive_mode()
