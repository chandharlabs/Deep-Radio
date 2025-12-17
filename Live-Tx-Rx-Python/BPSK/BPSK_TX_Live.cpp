#include <SoapySDR/Device.hpp>
#include <SoapySDR/Formats.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <thread>
#include <chrono>
#include <cstdio>

#ifdef _WIN32
#include <conio.h>
#else
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#endif

using namespace std;

vector<int> string_to_bits(const string &input) {
    vector<int> bits = {0, 1, 0, 1, 1, 1}; // Preamble
    for (char c : input) {
        for (int i = 7; i >= 0; --i) {
            bits.push_back((c >> i) & 1);
        }
    }
    return bits;
}

vector<complex<float>> generate_bpsk_symbols(const vector<int> &bits) {
    vector<complex<float>> symbols;
    for (int bit : bits) {
        symbols.emplace_back(bit == 1 ? 1.0f : -1.0f, 0.0f);
    }
    return symbols;
}

vector<complex<float>> upsample(const vector<complex<float>> &symbols, int factor) {
    vector<complex<float>> upsampled;
    for (const auto &sym : symbols) {
        for (int i = 0; i < factor; ++i) {
            upsampled.push_back(sym);
        }
    }
    return upsampled;
}

class BPSKTransmitter {
private:
    SoapySDR::Device *sdr;
    SoapySDR::Stream *txStream;
    double sampleRate;
    double symbolPeriod;
    int samplesPerSymbol;
    vector<complex<float>> upsampledSymbols;

public:
    BPSKTransmitter(double sr = 1e6, double symPer = 0.01, double freq = 700e6, double gain = 64.0) {
        sampleRate = sr;
        symbolPeriod = symPer;
        samplesPerSymbol = static_cast<int>(sampleRate * symbolPeriod);

        sdr = SoapySDR::Device::make("driver=lime");
        if (!sdr) {
            cerr << "Error: Could not create SDR device.\n";
            exit(1);
        }

        sdr->setSampleRate(SOAPY_SDR_TX, 0, sampleRate);
        sdr->setFrequency(SOAPY_SDR_TX, 0, freq);
        sdr->setGain(SOAPY_SDR_TX, 0, gain);

        txStream = sdr->setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, {0});
        sdr->activateStream(txStream);
    }

    void update_text(const string &text_in) {
        string text = "~" + text_in + "#";
        vector<int> rawBits = string_to_bits(text);
        vector<complex<float>> symbols = generate_bpsk_symbols(rawBits);
        upsampledSymbols = upsample(symbols, samplesPerSymbol);

        cout << "\n[Updated Transmission]" << endl;
        cout << "Text:       " << text << endl;
        cout << "Bitstream:  ";
        for (size_t i = 0; i < min<size_t>(rawBits.size(), 64); ++i) cout << rawBits[i];
        if (rawBits.size() > 64) cout << "...";
        cout << "\nSymbols:    ";
        for (size_t i = 0; i < min<size_t>(10, symbols.size()); ++i) cout << symbols[i] << " ";
        cout << "... (total " << symbols.size() << " symbols)\n" << endl;
    }

    void transmit_once() {
        size_t num_tx = upsampledSymbols.size();
        void *buffs[] = {upsampledSymbols.data()};
        int ret = sdr->writeStream(txStream, buffs, num_tx);
        if (ret < 0) {
            cerr << "Error writing stream: " << SoapySDR::errToStr(ret) << endl;
        }
        this_thread::sleep_for(chrono::microseconds(static_cast<int>(symbolPeriod * 1e6 * num_tx / samplesPerSymbol)));
    }

    void close() {
        sdr->deactivateStream(txStream);
        sdr->closeStream(txStream);
        SoapySDR::Device::unmake(sdr);
        cout << "Transmission stopped." << endl;
    }
};

int main() {
    BPSKTransmitter tx;

    string input;
    while (true) {
        cout << "Enter string to transmit once (or 'q' to quit): ";
        getline(cin, input);
        if (input == "q" || input == "Q") break;
        tx.update_text(input);
        tx.transmit_once();
        cout << "Transmission completed.\n" << endl;
    }

    tx.close();
    return 0;
}
