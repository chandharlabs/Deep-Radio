
// ---------- 2. bpsk_decoder.cc ----------

#include "bpsk_decoder.h"
#include <gnuradio/io_signature.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

namespace custom {

bpsk_decoder::bpsk_decoder(int transmittedBitsLen,
                           float transition_threshold,
                           int silence_threshold,
                           int bit_length,
                           int buffer_size,
                           const std::string& starting_sequence)
    : gr::sync_block("bpsk_decoder",
                     gr::io_signature::make(1, 1, sizeof(std::complex<float>)),
                     gr::io_signature::make(0, 0, 0)),
      d_transmitted_bits_len(transmittedBitsLen),
      d_transition_threshold(transition_threshold),
      d_silence_threshold(silence_threshold),
      d_bit_length(bit_length),
      d_bit_offset(30),
      d_buffer_size(buffer_size)
{
    std::stringstream ss(starting_sequence);
    std::string item;
    while (std::getline(ss, item, ','))
        d_starting_sequence.push_back(std::stoi(item));
}

bpsk_decoder::sptr bpsk_decoder::make(int transmittedBitsLen,
                                      float transition_threshold,
                                      int silence_threshold,
                                      int bit_length,
                                      int buffer_size,
                                      const std::string& starting_sequence)
{
    return gnuradio::get_initial_sptr(new bpsk_decoder(transmittedBitsLen,
                                                       transition_threshold,
                                                       silence_threshold,
                                                       bit_length,
                                                       buffer_size,
                                                       starting_sequence));
}

std::vector<std::string> bpsk_decoder::extract_messages(const std::string& data)
{
    std::vector<std::string> messages;
    size_t start = data.find('~');
    while (start != std::string::npos) {
        size_t stop = data.find('#', start);
        if (stop != std::string::npos) {
            messages.push_back(data.substr(start + 1, stop - start - 1));
            start = data.find('~', stop);
        } else {
            break;
        }
    }
    return messages;
}

void bpsk_decoder::process_data(const std::vector<std::complex<float>>& data)
{
    std::string message = "";
    const int half_bit = d_bit_length / 2;
    std::vector<float> real(data.size());
    for (size_t i = 0; i < data.size(); i++)
        real[i] = std::real(data[i]);

    for (size_t i = 0; i + d_transmitted_bits_len * d_bit_length < real.size(); i += d_silence_threshold) {
        std::vector<int> bits;
        for (int b = 0; b < d_transmitted_bits_len; ++b) {
            size_t idx = i + b * d_bit_length + d_bit_offset;
            if (idx >= real.size()) break;
            float val = real[idx];
            bits.push_back(val > d_transition_threshold ? 1 : (val < -d_transition_threshold ? 0 : -1));
        }

        if (bits.size() == d_transmitted_bits_len && bits[3] != -1) {
            if (bits[3] == 0 && bits[4] == 0 && bits[5] == 0) {
                for (int& bit : bits) bit = 1 - bit;
            }

            if (std::equal(bits.begin() + 3, bits.begin() + 6, d_starting_sequence.begin() + 3)) {
                std::vector<int> payload(bits.begin() + d_starting_sequence.size(), bits.end());
                for (size_t i = 0; i + 8 <= payload.size(); i += 8) {
                    std::string byte_str;
                    for (int j = 0; j < 8; ++j)
                        byte_str += std::to_string(payload[i + j]);
                    char c = static_cast<char>(std::stoi(byte_str, nullptr, 2));
                    message += c;
                }
            }
        }
    }

    std::vector<std::string> messages = extract_messages(message);
    std::ofstream out("decoded_message.txt", std::ios::app);
    for (const std::string& m : messages)
        out << m << std::endl;
}

int bpsk_decoder::work(int noutput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
{
    const std::complex<float>* in = reinterpret_cast<const std::complex<float>*>(input_items[0]);
    std::vector<std::complex<float>> new_data(in, in + noutput_items);

    std::lock_guard<std::mutex> lock(d_mutex);
    d_data_buffer.insert(d_data_buffer.end(), new_data.begin(), new_data.end());

    if (d_data_buffer.size() >= (size_t)d_buffer_size) {
        std::vector<std::complex<float>> data_to_process(d_data_buffer.begin(), d_data_buffer.begin() + d_buffer_size);
        d_data_buffer.erase(d_data_buffer.begin(), d_data_buffer.begin() + d_buffer_size);
        process_data(data_to_process);
    }

    return noutput_items;
}

} // namespace custom
