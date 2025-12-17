import numpy as np
import itertools
from datetime import datetime
import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import *

# from scipy import signal
from scipy.fftpack import fftshift
from contextlib import closing
import sys
from py3gpp.configs.nrCarrierConfig import nrCarrierConfig
from py3gpp.nrPBCH import nrPBCH
from py3gpp.nrPBCHIndices import nrPBCHIndices
from py3gpp.nrPDSCHDMRSIndices import nrPDSCHDMRSIndices
from py3gpp.nrPBCHDMRS import nrPBCHDMRS
from py3gpp.nrPDSCH import nrPDSCH
from py3gpp.nrPDSCHIndices import nrPDSCHIndices
from py3gpp.nrPSS import nrPSS
from py3gpp.nrPSSIndices import nrPSSIndices
from py3gpp.nrSSS import nrSSS
from py3gpp.nrSSSIndices import nrSSSIndices
from py3gpp.nrResourceGrid import nrResourceGrid
from py3gpp.nrPDSCHDMRS import PDSCHDMRSSyms
from py3gpp.configs.nrPDSCHConfig import nrPDSCHConfig
from py3gpp.configs.nrCarrierConfig import nrCarrierConfig
from py3gpp.nrOFDMModulate import nrOFDMModulate
from py3gpp.nrOFDMDemodulate import nrOFDMDemodulate
from test_data.pdsch import pdsch_symbols_ref, pdsch_bits_ref

final_sequence_data = None


def nrPBCHDMRSIndicesU(ncellid):
    indices_sym1_3 = np.arange(ncellid % 4, 240, 4)
    indices_sym2 = np.concatenate(
        (np.arange(ncellid % 4, 48, 4), np.arange(192 + ncellid % 4, 240, 4))
    )
    return (indices_sym1_3, indices_sym2, indices_sym1_3)


def tempnrPBCHIndicesU(ncellid):
    indices_sym1_3 = np.arange(0, 240, 1)
    indices_sym2 = np.concatenate((np.arange(0, 48, 1), np.arange(192, 240, 1)))
    return (indices_sym1_3, indices_sym2, indices_sym1_3)


def overlap_add(S, S_t, WinSize):
    WinSize = int(WinSize)
    if WinSize > 0:
        retval = np.concatenate(
            (S[:-WinSize], S[-WinSize:] + S_t[:WinSize], S_t[WinSize:])
        )
    else:
        retval = np.concatenate((S, S_t))
    return retval


def ofdm_dl_waveform_5g(
    NFFT, NumOfSymbols, mu, Nrb, Guard, WinSize, cpsize, ak_indices, tag, modulation
):
    # NFFT = 1024  # NFFT fixed in 5G
    NoOfCarriers = 12 * Nrb  # Num Carriers/Tones = 12 * Num Res Blocks
    deltaF = 2**mu * 15000  # sub-carrier spacing 2^mu * 15KHz
    CPSize = cpsize  # Cyclic Prefix is 288
    CPSize_FirstSymbol = 320  # First Symbol of half SubFrame Cyc Prefix is 320
    subFrameDur = 1 / 1000  # 1msec
    FrameDur = 10 * subFrameDur  # 10 msec
    T_symbol = 1 / deltaF  # OFDM Symbol Duration without CP
    deltaF_max = 15000 * 2**5  # 480KHz
    deltaF_ref = 15000  # LTE sub-carrier spacing fixed to 15KHz
    NFFT_ref = 2048  # NFFT fixed in LTE
    # Derived Params
    Bandwidth = deltaF * NoOfCarriers  # Sub-Carrier spacing * num Sub-Carriers
    kappa = (deltaF_max * NFFT) / (
        deltaF_ref * NFFT_ref
    )  # Constant : max Number 5G symbols possible in LTE Symbol Duration
    Nu = NFFT_ref * kappa / (2**mu)  # Num of OFDM samples in 32 symbols
    Ncpmu = 144 * kappa / (2**mu)  # total CP samples in 32 symbols
    Ts = (
        1 / deltaF_max / NFFT
    )  # symbolDur if Symbol occupies 480*4096 KHz (Max bandwidth possible in 5G NR with 4096 tones)
    Fs = NFFT / T_symbol  # Sampling Frequency to achieve desired Symbol Duration
    Fc = Bandwidth / 2 - Guard * deltaF  # Cut-off frequency for LPF FIR design
    t = np.linspace(
        0, (Nu + Ncpmu) * Ts, NFFT + CPSize, endpoint=False
    )  # NFFT+CP size time-domain sequence required 0<=t<(Nu+Ncpmu)Ts
    data = np.loadtxt(
        "modData.txt"
    )  # BPSK modulated random data stored in 'modData.txt'
    NumData = (
        NoOfCarriers - 2 * Guard - 1
    )  # Num of nPSK symbols which can be loaded on 1 OFDM Symbol
    # Window design for WOLA operation to suppress out of band spectral leakage
    if WinSize > 1:
        alpha = 0.5
        nFilt = np.linspace(-np.pi / 2, 0, WinSize)
        x_win = np.multiply(np.cos(nFilt), np.cos(nFilt))
    # Below Loop will run for Number of OFDM symbols to be generated
    for num in range(0, NumOfSymbols):
        # a_k is the sequence of nPSK symbols which will be loaded on Sub-carriers(FFT Tones)
        PBCHDMRSIndices1, PBCHDMRSIndices2, PBCHDMRSIndices3 = nrPBCHDMRSIndicesU(1)

        pBCHDMRS = nrPBCHDMRS(1, 0)
        pBCHDMRS1 = pBCHDMRS[0 : len(PBCHDMRSIndices1)]
        pBCHDMRS2 = pBCHDMRS[
            len(PBCHDMRSIndices1) : len(PBCHDMRSIndices1) + len(PBCHDMRSIndices2)
        ]
        pBCHDMRS3 = pBCHDMRS[len(pBCHDMRS) - len(PBCHDMRSIndices1) : len(pBCHDMRS)]

        tempPBCHIndices1, tempPBCHIndices2, tempPBCHIndices3 = tempnrPBCHIndicesU(1)

        PBCHIndices1 = np.setdiff1d(tempPBCHIndices1, PBCHDMRSIndices1)
        PBCHIndices2 = np.setdiff1d(tempPBCHIndices2, PBCHDMRSIndices2)
        PBCHIndices3 = np.setdiff1d(tempPBCHIndices3, PBCHDMRSIndices3)

        jj = np.ones((864, 1), dtype=int)
        pBCH = nrPBCH(1, 0, np.array(list(itertools.chain(*jj)), dtype=(int)))
        pBCH1 = pBCH[0 : len(PBCHIndices1)]
        pBCH2 = pBCH[len(PBCHIndices1) : len(PBCHIndices1) + len(PBCHIndices2)]
        pBCH3 = pBCH[len(pBCH) - len(PBCHIndices1) : len(pBCH)]

        pdsch = nrPDSCHConfig()
        carrier = nrCarrierConfig()
        carrier.NSizeGrid = 10
        pdsch.NSizeBWP = 2
        pdsch.NStartBWP = 1
        pdsch.PRBSet = np.arange(pdsch.NSizeBWP)
        pdschIndices = nrPDSCHIndices(carrier, pdsch)

        if tag == "PSS":
            a_k = np.zeros((NFFT, 1), dtype=complex)
            pSS = nrPSS(1)
            temp = 0
            # print("PSS : ", nrPSSIndices())
            for item in nrPSSIndices():                
                a_k[int(item - 119 + (Nrb * 12/2))] = pSS[temp]
                print(int(item -119 + (Nrb * 12/2)))
                temp = temp + 1

        elif tag == "SSS":
            a_k = np.zeros((NFFT, 1), dtype=complex)
            sSS = nrSSS(1)
            temp = 0
            for item in nrSSSIndices():
                a_k[int(item - 119 + (Nrb * 12/2))] = sSS[temp]
                temp = temp + 1

        elif tag == "PBCH1":  # "PBCH1+PBCHDMRS1"
            a_k = np.zeros((NFFT, 1), dtype=complex)
            temp = 0
            for item in PBCHIndices1:
                a_k[int(item - 119 + (Nrb * 12/2))] = pBCH1[temp]
                temp = temp + 1
            # temp = 0
            # for item in PBCHDMRSIndices1:
            #     a_k[item] = pBCHDMRS1[temp]
            #     temp = temp + 1

        elif tag == "SSS+PBCH2":  # "SSS+PBCH2+PBCHDMRS2"
            a_k = np.zeros((NFFT, 1), dtype=complex)
            sSS = nrSSS(1)
            temp = 0
            for item in nrSSSIndices():
                a_k[int(item - 119 + (Nrb * 12/2))] = sSS[temp]
                temp = temp + 1
            temp = 0
            for item in PBCHIndices2:
                a_k[int(item - 119 + (Nrb * 12/2))] = pBCH2[temp]
                temp = temp + 1
            # temp = 0
            # for item in PBCHDMRSIndices2:
            #     a_k[item] = pBCHDMRS2[temp]
            #     temp = temp + 1

        elif tag == "PBCH3":  # "PBCH3+PBCHDMRS3"
            a_k = np.zeros((NFFT, 1), dtype=complex)
            temp = 0
            for item in PBCHIndices3:
                a_k[int(item - 119 + (Nrb * 12/2))] = pBCH3[temp]
                temp = temp + 1
            # temp = 0
            # for item in PBCHDMRSIndices3:
            #     a_k[item] = pBCHDMRS3[temp]
            #     temp = temp + 1

        elif tag == "PDSCH":
            print("This is not working, need to fix this.")
            a_k = np.zeros((NFFT, 1), dtype=complex)
            temp = 0
            print("PDSCH length : ", len(pdschIndices))
            for item in pdschIndices:
                a_k[item] = 1
                temp = temp + 1
            print("PDSCH Indices : ", pdschIndices)

        elif tag == "SPECIFIC":
            a_k = ak_indices
            
        elif tag == "CUSTOM":
            ak_temp = [1, -1, -1]
            ak_indices = [300, 302, 304]

            a_k = np.zeros((NFFT, 1), dtype=complex)
            temp = 0
            for item in ak_indices:
                a_k[item] = ak_temp[temp]
                temp = temp + 1

        elif tag == "BLOCK":
            a_k = np.zeros((NFFT, 1), dtype=complex)
            for item in ak_indices:
                if modulation == "bpsk":
                    a_k[item] = np.random.choice([-1, 1])
                elif modulation == "qpsk":
                    a_k[item] = np.random.choice(
                        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
                    ) / np.sqrt(2)
                elif modulation == "16qam":
                    a_k[item] = np.random.choice(
                        [
                            1 + 1j,
                            1 + 3j,
                            3 + 1j,
                            3 + 3j,
                            1 - 1j,
                            1 - 3j,
                            3 - 1j,
                            3 - 3j,
                            -1 + 1j,
                            -1 + 3j,
                            -3 + 1j,
                            -3 + 3j,
                            -1 - 1j,
                            -1 - 3j,
                            -3 - 1j,
                            -3 - 3j,
                        ]
                    ) / np.sqrt(10)
                elif modulation == "64qam":
                    a_k[item] = np.random.choice(
                        [
                            1 + 1j,
                            1 + 3j,
                            1 + 5j,
                            1 + 7j,
                            3 + 1j,
                            3 + 3j,
                            3 + 5j,
                            3 + 7j,
                            5 + 1j,
                            5 + 3j,
                            5 + 5j,
                            5 + 7j,
                            7 + 1j,
                            7 + 3j,
                            7 + 5j,
                            7 + 7j,
                            1 - 1j,
                            1 - 3j,
                            1 - 5j,
                            1 - 7j,
                            3 - 1j,
                            3 - 3j,
                            3 - 5j,
                            3 - 7j,
                            5 - 1j,
                            5 - 3j,
                            5 - 5j,
                            5 - 7j,
                            7 - 1j,
                            7 - 3j,
                            7 - 5j,
                            7 - 7j,
                            -1 + 1j,
                            -1 + 3j,
                            -1 + 5j,
                            -1 + 7j,
                            -3 + 1j,
                            -3 + 3j,
                            -3 + 5j,
                            -3 + 7j,
                            -5 + 1j,
                            -5 + 3j,
                            -5 + 5j,
                            -5 + 7j,
                            -7 + 1j,
                            -7 + 3j,
                            -7 + 5j,
                            -7 + 7j,
                            -1 - 1j,
                            -1 - 3j,
                            -1 - 5j,
                            -1 - 7j,
                            -3 - 1j,
                            -3 - 3j,
                            -3 - 5j,
                            -3 - 7j,
                            -5 - 1j,
                            -5 - 3j,
                            -5 - 5j,
                            -5 - 7j,
                            -7 - 1j,
                            -7 - 3j,
                            -7 - 5j,
                            -7 - 7j,
                        ]
                    ) / np.sqrt(42)
                elif modulation == "8psk":
                    a_k[item] = np.exp(1j * (np.pi / 4) * np.random.choice(range(8)))
                else:
                    a_k[item] = 1 #np.random.choice([-1, 1])

        # k is sub-carrier index starting from most negative tone (I am generating DC centered OFDM Spectrum)
        k = np.linspace(-int(NoOfCarriers / 2), int(NoOfCarriers / 2) - 1, NoOfCarriers)
        # S_t is time-domain OFDM symbol with CP. Above loop generates OFDM Symbol with CP appended already :)
        S_t = np.zeros(t.size)
        for i in k:
            S_t = (
                a_k[int(i + NoOfCarriers / 2)]
                * np.exp(1j * 2 * np.pi * i * deltaF * (t - Ncpmu * Ts))
                + S_t
            )
        # Apply windowing (WOLA) only when window size is greater than 1
        if WinSize > 1:
            S_t = np.concatenate((S_t, S_t[CPSize : CPSize + WinSize]))
            # S_t = np.convolve(S_t, x_win, mode='same')/sum(x_win)
            S_t[:WinSize] = S_t[:WinSize] * x_win
            S_t[-WinSize:] = S_t[-WinSize:] * np.flip(x_win)
        if num == 0:
            S = S_t
        else:
            S = overlap_add(S, S_t, WinSize)
    if WinSize > 1:
        x1 = np.fft.fft(S[-4096 - WinSize : -WinSize], 4096) / 4096
    else:
        x1 = np.fft.fft(S[-4096:], 4096) / 4096

    return S


def generate_dynamic_sequence(
    NFFT,
    sequence,
    NumOfSymbols,
    mu,
    Nrb,
    Guard,
    WinSize,
    cpsize,
    numFrames,
    total_length=140,
):
    waveform = np.array([], dtype=complex)

    repeat_allowed = ["PSS", "SSS", "PBCH1", "SSS+PBCH2", "PBCH3"]
    custom_blocks = ["PDSCH", "PDCCH"]
    symbol = 0
    # Iterate over the sequence to generate the waveform
    frames = numFrames
    for frame in range(0, frames):
        for index in range(0, total_length):
            # Find the matching tuple for the current index
            tupple = next((t for t in sequence if t[1] == index), None)

            if (
                tupple
                and (tupple[0] in custom_blocks)
                and tupple[1] == index
                and tupple[2] == frame
            ):
                ak_indices = range(tupple[3], tupple[3] + tupple[5])  # range(j*1 + 1, j*36 + 37)
                # a_k = np.zeros((NFFT, 1), dtype=complex)
                # print("Ak length : ", len(ak_indices))
                # # for j in range(0, tupple[4]):
                #     # index = index + 1
                # for idx in ak_indices:
                #     a_k[idx] = 1 + j

                for i in range(0, tupple[4]):
                    # Generate the OFDM waveform for the given tag
                    symbol = ofdm_dl_waveform_5g(
                        NFFT,
                        NumOfSymbols,
                        mu,
                        Nrb,
                        Guard,
                        WinSize,
                        cpsize,
                        ak_indices=ak_indices,
                        tag="BLOCK",
                        modulation=tupple[6],
                    )
                    waveform = np.append(waveform, symbol)
                    # index = index +1
            elif (
                tupple
                and (tupple[0] == "CUSTOM")
                and tupple[1] == index
                and tupple[2] == frame
            ):
                ak_indices = [1, 5, 10, 20, 25, 50, 55, 60, 65]
                print("Ak Indices : ", ak_indices)

                for i in range(0, tupple[4]):
                    # Generate the OFDM waveform for the given tag
                    symbol = ofdm_dl_waveform_5g(
                        NFFT,
                        NumOfSymbols,
                        mu,
                        Nrb,
                        Guard,
                        WinSize,
                        cpsize,
                        ak_indices=ak_indices,
                        tag="CUSTOM",
                        modulation=tupple[6],
                    )
                    waveform = np.append(waveform, symbol)
                    # index = index +1
            elif tupple and tupple[0] in repeat_allowed:
                ak_indices = range(0, 2)
                # Generate the OFDM waveform for the given tag
                symbol = ofdm_dl_waveform_5g(
                    NFFT,
                    NumOfSymbols,
                    mu,
                    Nrb,
                    Guard,
                    WinSize,
                    cpsize,
                    ak_indices,
                    tag=tupple[0],
                    modulation=None,
                )
            else:
                # Generate zeros if no tag is found for the current index
                symbol = np.zeros((NFFT + cpsize,), dtype=complex)

            # Append the generated symbol to the waveform
            waveform = np.append(waveform, symbol)

        # Add remaining zeros to make the length equal to total_length
        remaining_length = total_length - len(waveform) // NumOfSymbols
        if remaining_length > 0:
            waveform = np.append(
                waveform, np.zeros((remaining_length * NumOfSymbols,), dtype=complex)
            )

    return waveform


def generate_signal(
    NFFT,
    num_of_symbols,
    mu,
    nrb,
    guard,
    win_size,
    total_length,
    sequence,
    cpsize,
    numFrames,
):
    global final_sequence_data

    print("Sequence : ", sequence)
    final_sequence_data = generate_dynamic_sequence(
        NFFT,
        sequence,
        num_of_symbols,
        mu,
        nrb,
        guard,
        win_size,
        cpsize,
        numFrames,
        total_length,
    )  # generate the pattern
    x = np.zeros((5 * NFFT + cpsize,), dtype=complex)
    # Store the data as iq
    final_sequence_data = np.append(x, final_sequence_data)

    final = final_sequence_data.astype(np.complex64)

    filename = "IQoutput/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".iq"
    final.tofile(filename)
    return filename


def get_final_sequence_data():
    global final_sequence_data
    return final_sequence_data


def get_latest_filename():
    dir_path = "IQoutput"
    files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".iq")
    ]
    if files:
        files.sort(key=os.path.getctime)
        return os.path.abspath(files[-1])
    return None
