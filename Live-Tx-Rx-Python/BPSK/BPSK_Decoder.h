// ---------- 1. bpsk_decoder.h ----------

#ifndef INCLUDED_CUSTOM_BPSK_DECODER_H
#define INCLUDED_CUSTOM_BPSK_DECODER_H

#include <gnuradio/sync_block.h>
#include <vector>
#include <complex>
#include <mutex>

namespace custom {

class bpsk_decoder : public gr::sync_block
{
private:
    int d_transmitted_bits_len;
    float d_transition_threshold;
    int d_silence_threshold;
    int d_bit_length;
    int d_bit_offset;
    int d_buffer_size;
    std::vector<int> d_starting_sequence;
    std::vector<std::complex<float>> d_data_buffer;
    std::vector<std::vector<int>> d_bit_sequences;
    std::mutex d_mutex;

    void process_data(const std::vector<std::complex<float>>& data);
    std::vector<std::string> extract_messages(const std::string& data);

public:
    typedef boost::shared_ptr<bpsk_decoder> sptr;

    static sptr make(int transmittedBitsLen = 19,
                     float transition_threshold = 0.5,
                     int silence_threshold = 30000,
                     int bit_length = 1300,
                     int buffer_size = 500000,
                     const std::string& starting_sequence = "0,1,0,1,1,1");

    bpsk_decoder(int transmittedBitsLen,
                 float transition_threshold,
                 int silence_threshold,
                 int bit_length,
                 int buffer_size,
                 const std::string& starting_sequence);

    int work(int noutput_items,
             gr_vector_const_void_star &input_items,
             gr_vector_void_star &output_items);
};

} // namespace custom

#endif // INCLUDED_CUSTOM_BPSK_DECODER_H
