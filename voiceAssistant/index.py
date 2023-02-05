import pyttsx3  # download -> pip install pyttsx3
import pyjokes  # download -> pip install pyjokes
import speech_recognition as sr  # download -> pip install speechrecognition
import pyaudio  # download
import webbrowser
import datetime
import os
import time


def sptext():
    recognizer = sr.Recognizer()  # catch the voice from the microphone
    with sr.Microphone() as source:
        while True:
            print("Listening...")
            # noice cancellation
            recognizer.adjust_for_ambient_noise(source)
            # source data
            audio = recognizer.listen(source)
            # read the data ( if not read then it gives error)
            try:
                print("Recognizing...")
                data = recognizer.recognize_google(audio)
                # checking for data(voice) is empty or not
                print(data, "data voice")
                if (data):
                    return data
                else:
                    continue
            except sr.UnknownValueError:
                print("Not understand")


def speechtx(x):
    engine = pyttsx3.init()  # invoke the object
    # voice - (male or female)
    voices = engine.getProperty('voices')
    # get only one voice -> 0 for male and 1 for female
    engine.setProperty('voice', voices[0].id)
    # speed
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 150)

    engine.say(x)  # voice data
    engine.runAndWait()


if __name__ == '__main__':

    # voice assistant name
    assistantName = "hey shivam"

    if assistantName in sptext().lower():

        while True:
            data1 = sptext().lower()

            if "your name" in data1:
                name = "my name is shivam"
                speechtx(name)

            elif "old are you" in data1:
                age = "I am twenty years old"
                speechtx(age)

            elif 'what is the time right now' in data1:
                time = datetime.datetime.now().strftime("%I%M%p")
                speechtx(time)

            elif 'open youtube' in data1:
                webbrowser.open("https://www.youtube.com/")

            elif 'ek joke suna de bhai' in data1:
                joke = pyjokes.get_joke(language="en", category="neutral")
                speechtx(joke)

            elif 'play video' in data1:
                address = 'S:\\Programming Language\\Java\\video\\playlist'
                listvideo = os.listdir(address)
                os.startfile(os.path.join(address, listvideo[0]))

            elif 'exit' in data1:
                speechtx('thank you')
                break

            time.sleep(2)

    else:
        print("thanks")
