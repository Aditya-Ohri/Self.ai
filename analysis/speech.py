import speech_recognition as sr


class SpeechListener:

    def __init__(self, language: str='en'):
        self._language = language
        self.speech = '<NO SPEECH REGISTERED>'

    def listen(self) -> str:
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                human = str(recognizer.recognize_google(audio, language=self._language))
        except sr.UnknownValueError:
            human = '<NO SPEECH REGISTERED>'

        return human

    def change_language(self, language: str):
        self._language = language


#listener = SpeechListener()
#print(listener.listen())
