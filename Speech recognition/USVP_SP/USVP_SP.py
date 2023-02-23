import requests
from requests.auth import HTTPDigestAuth
from IPython.display import Audio
from collections import namedtuple
from playsound import playsound
import librosa
import numpy as np
import math
from scipy.io.wavfile import write
import re

# Download data
from urllib.request import urlretrieve
import os

# TTS-Service credentials
USERNAME = "kky_usvp"
PASSWORD = "queN7aex"

# TTS-Service API ROOT and available Czech voices and formats
TTS_ROOT = "https://services.speechtech.cz/tts/v3/synth"
VOICES_CS = ["Jan210", "Alena210", "Iva210", "Radka210", "Stanislav210", "Jiri210"]
FORMATS = ["mp3", "wav44kHz16bit"]

# UWebASR API ROOT
ASR_ROOT = "https://uwebasr.zcu.cz/api/v1/"

def synthesize(out_fn, text, voice, fmt):
    """Synthesize the text into out_fn

    The voice of speech synthesis and the output format could be specified.

    Example:
    synthesize("test.wav", "hello world", "Jeremy210", "wav44kHz16bit")

    TTS-Server is provided by the company SpeechTech s.r.o.
    """
    r = requests.get(TTS_ROOT,
                     auth=HTTPDigestAuth(USERNAME, PASSWORD),
                     params={"text": text,
                             "engine": voice,
                             "format": fmt}
                    )
    r.raise_for_status()
    with open(out_fn, "wb") as fw:
        fw.write(r.content)

def recognize(fn, model, words_only=False):
    """Recognizes the file fn using the UWebASR service

    The model must be supplied to identify the recognition model.

    If words_only is True, then the function returns array of words,
    otherwise it returns array of dictionaries containing more detailed
    recognition results.

    Example: 
    recognize("test.wav", "CLARIN_ASR/CZ")

    UWebASR service is provided by Department of Cybernetics, University of
    West Bohemia
    """
    with open(fn, "rb") as fr:
        r = requests.post(ASR_ROOT+model, data=fr, params={"format": "json"})
        r.raise_for_status()
        ret = r.json()
    if words_only:
        ret = [i["word"] for i in ret]
    return ret

def download(url, file):
    if not os.path.isfile(file):
        print("Download file... " + file + " ...")
        urlretrieve(url,file)
        print("File downloaded")

def loadTXTFile(fileName):
    with open(fileName, 'r', encoding='utf-8' ) as file:
        lines = file.read()
    file.close()
    return lines

synthesize("test.wav", "Ahoj, jak se máš?", "Jiri210", "wav44kHz16bit")
#Audio("test.wav")
#playsound("test.wav")

recognize("test.wav", "CLARIN_ASR/CZ", words_only=True)

OP_OK = 'o'
OP_SUB = 's'
OP_INS = 'i'
OP_DEL = 'd'

COST_SUB = 1
COST_INS = 1
COST_DEL = 1

def Levenshtein(vzor, test):
    """Calculates Levenshtein distance and corresponding edit operations"""
    m = len(vzor)
    n = len(test)
    vzor = [' ']+vzor
    test = [' ']+test

    last_row = [i*COST_INS for i in range(n+1)]
    last_row_op = [OP_INS*i for i in range(n+1)]

    for i in range(1, m+1):
        current = [i*COST_DEL]
        current_op = [OP_DEL*i]
        for j in range(1, n+1):
            if vzor[i] == test[j]:
                cost_sub = 0
                sub_op = OP_OK
            else:
                cost_sub = COST_SUB
                sub_op = OP_SUB
            cost_del = COST_INS
            cost_ins = COST_DEL
            min_cost, min_op = min(
                    [last_row[j]+cost_del, last_row_op[j]+OP_DEL],
                    [current[j-1]+cost_ins, current_op[j-1]+OP_INS],
                    [last_row[j-1]+cost_sub, last_row_op[j-1]+sub_op],
            )

            current.append(min_cost)
            current_op.append(min_op)
        last_row = current
        last_row_op = current_op

    return current[-1], current_op[-1]

edit_counts = namedtuple("edit_counts", "N S D I H")

def calc_edit_ops(ref, hyp):
    """Calculates and returns number of edit operations

    N Number of tokens in the reference
    H Number of correctly recognized tokens
    S Number of substitutions
    D Number of deletions
    I Number of insertions
    """
    dist, ops = Levenshtein(ref, hyp)
    H = I = D = S = 0
    for op in ops:
        if op == OP_SUB:
            S += 1
        elif op == OP_DEL:
            D += 1
        elif op == OP_INS:
            I += 1
        elif op == OP_OK:
            H += 1
    N = len(ref)
    return edit_counts(N=N, S=S, D=D, I=I, H=H)

e = calc_edit_ops(["ahoj", "jak", "se", "máš"], ["ahojky", "jak", "se"])

#print(e)
accuracy = (e.N-e.S-e.D-e.I)/e.N
#print(accuracy)

INPUT_TEXT = """Provozovatel holešovské autoškoly Vladimír Dohnal popisuje rozhodnutí vlády o dřívějším otevření některých obchodů a provozoven za šílené a nepřipravené.
Kvůli šibeničnímu termínu tak v pondělí neotevřely některé zoologické zahrady či knihovny, byť už by mohly.
Přes víkend totiž většinou nestačily zajistit hygienická opatření nebo spustit online nákup vstupenek.
Regionální stanice Českého rozhlasu těžkosti některých provozovatelů zmapovaly.
"""

def deleteInterpunction(text):
    textBezInterpunkce = re.sub(r'[^\w\s]', '', text)
    listTextBezInterpunkce = list(textBezInterpunkce.split(" "))
    while("" in listTextBezInterpunkce):
        listTextBezInterpunkce.remove("")
    while("\n" in listTextBezInterpunkce):
        listTextBezInterpunkce.remove("\n")
    return listTextBezInterpunkce

def zpracujBlokove(data_string, record_name, voice, printRecognitionString):
    synthesize(record_name, data_string, voice, "wav44kHz16bit")
    #Audio("data_String.wav")
    #playsound("data_String.wav")
    recognition_string = recognize(record_name, "CLARIN_ASR/CZ", words_only=True)
    if (printRecognitionString):
        print(recognition_string)
    listDataString = deleteInterpunction(data_string)
    e = calc_edit_ops(listDataString, recognition_string)
    #print(e)
    accuracy = (e.N-e.S-e.D-e.I)/e.N
    print(f'Accuracy: ' + ("{:.3f}".format(accuracy)))

def zpracujZasumenouNahravku(data_string, record_noise_name, noise_type, intensity):
    recognition = recognize(record_noise_name, "CLARIN_ASR/CZ", words_only=True)
    #print(listToString(recognition))
    listDataString = deleteInterpunction(data_string)
    e = calc_edit_ops(listDataString, recognition)
    #print(e)
    accuracy = (e.N-e.S-e.D-e.I)/e.N
    print(f'Accuracy ({noise_type} with intensity = {intensity}): ' + ("{:.2f}".format(accuracy)))

def zpracujPoVetach(data_sentences, voice):
    accuracyList = []
    e_N = 0
    e_S = 0
    e_D = 0
    e_I = 0
    for i in range(len(data_sentences)):
        recordName = "veta" + str((i+1)) + ".wav"
        veta = data_sentences[i]
        veta_length = len(veta)
        synthesize(recordName, data_sentences[i], voice, "wav44kHz16bit")
        #Audio(recordName)
        #playsound(recordName)
        recognition_sentence = recognize(recordName, "CLARIN_ASR/CZ", words_only=True)
        listVetaBezInterpunkce = deleteInterpunction(veta)
        e = calc_edit_ops(listVetaBezInterpunkce, recognition_sentence)
        accuracy = (e.N-e.S-e.D-e.I)/e.N
        accuracyList.append(accuracy)
        e_N += e.N
        e_S += e.S
        e_D += e.D
        e_I += e.I 
    accuracy = (e_N-e_S-e_D-e_I)/e_N 
    print("Zpracovani po vetach: ")
    for i in range(len(accuracyList)):
        print(f'Line {i+1}: ' + ("{:.3f}".format(accuracyList[i])))
    print(f'Weighted mean accuracy: ' + ("{:.3f}".format(accuracy)))
    print()

def get_white_noise(signal, intensity) :
    #RMS value of signal
    RMS_s = math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n = math.sqrt(RMS_s**2/(pow(10, intensity/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n = RMS_n
    white_noise = np.random.normal(0, STD_n, signal.shape[0])

    return white_noise

def get_white_noise_simply(signal):
    return (np.random.normal(0, .1, signal.shape))

def get_noise_from_sound(signal, noise, intensity):
    RMS_s = math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n = math.sqrt(RMS_s**2/(pow(10, intensity/10)))
    
    #current RMS of noise
    RMS_n_current = math.sqrt(np.mean((noise**2)))
    noise = noise*(RMS_n/RMS_n_current)
    
    return noise

def listToString(list): 
    str1 = " " 
    return (str1.join(list))

# Stažení a inicializace dat
filename_1_url = "http://home.zcu.cz/~honzas/KKY_USVP.10.txt"
filename_2_url = "http://home.zcu.cz/~honzas/KKY_USVP.20.txt"

filename1 = "KKY_USVP.10.txt"
filename2 = "KKY_USVP.20.txt"

download(filename_1_url, filename1)
download(filename_2_url, filename2)

data1 = loadTXTFile(filename1)
data2 = loadTXTFile(filename2)

# Ukol 1)
data1_sentences = data1.splitlines()
data1_String = data1

#zpracujPoVetach(data1_sentences, "Alena210")

recordName1 = "data_String_10.wav"
print("Blokove zpracovani souboru " + filename1 + ": " )
zpracujBlokove(data1_String, recordName1, "Alena210", printRecognitionString = False)
print()

# Ukol 2)
data2_string = data2
recordName2 = "data_String_20.wav"
print("Blokove zpracovani souboru " + filename2 + ": " )
#zpracujBlokove(data2_string, recordName2, "Alena210", printRecognitionString = False)

noiseRecordName = "zpravy.wav"

# Rozpoznavani vlastni nahravky 
#recognition = recognize("zpravy.wav", "CLARIN_ASR/CZ", words_only=True)
#textZpravy = listToString(recognition)
#print(textZpravy)

# Vytvoreni nahravek se sumem
signal, sr = librosa.load(recordName2)
signal_noise, sr_noise = librosa.load(noiseRecordName)

# Je pocitano s tim, ze sum je delsi nez signal, ke kterymu chci onen sum pridat
if(len(signal_noise) > len(signal)):
    signal_noise = signal_noise[0:len(signal)]

intensity_white_noise = 5 
white_noise = get_white_noise(signal, intensity_white_noise)
#white_noise = get_white_noise_simply(signal)
record_white_noise_name = "white_noise.wav"
signal_white_noise = signal + white_noise
write(record_white_noise_name, sr, signal_white_noise)
zpracujZasumenouNahravku(data2_string, record_white_noise_name, "white noise", intensity_white_noise)
print()

intensities = [0.1, 0.5, 1.0, 2.0]
for intensity in intensities:
    sound_noise = get_noise_from_sound(signal, signal_noise, intensity)
    signal_noise_from_sound = signal + sound_noise 
    record_sound_noise_name = "noise_from_sound_intensity_" + str(intensity) + ".wav"
    write(record_sound_noise_name, sr_noise, signal_noise_from_sound)
    zpracujZasumenouNahravku(data2_string, record_sound_noise_name, "sound noise", intensity)

print()

# Ukol 3)
slova = "nejnezpravděpodobnostňovávatelnějšími, Popocatépetl, USA, piercing, bouldering, snowboard, kyselina trihydrogenfosforečná"
print(slova)
zpracujBlokove(slova, "slova.wav", "Alena210", printRecognitionString = True)



