import spacy
from spacy.language import Language
from transformers import pipeline
from spacy.tokens import Doc
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DICTIONARY = {"lol": "laughing out loud", "brb": "be right back"}
DICTIONARY.update({value: key for key, value in DICTIONARY.items()})


@Language.factory("sentiment_analysis",
                  default_config={"sentiment_analysis_model": "siebert/sentiment-roberta-large-english", "gpu": "mps"})
def create_acronym_component(nlp: Language, name: str, sentiment_analysis_model: str, gpu: str):
    return SentimentAnalyser(nlp, name, sentiment_analysis_model, gpu)


class SentimentAnalyser:
    def __init__(self, nlp: Language, name: str, sentiment_analysis_model: str, gpu: str):
        # Create the matcher and match on Token.lower if case-insensitive
        self.model_string = sentiment_analysis_model
        self.name = name
        self.device = gpu
        if self.model_string == "vaderSentiment":
            self.analyser = SentimentIntensityAnalyzer()
        else:
            self.analyser = pipeline("sentiment-analysis", self.model_string, device=self.device)

    def __call__(self, doc: Doc) -> Doc:
        # Add the matched spans when doc is processed
        for i, sent in enumerate(doc.sents):
            if not doc.user_span_hooks.get(i, None):
                doc.user_span_hooks[i] = {}
            sentiment = None
            if self.model_string == "vaderSentiment":
                sentiment = self.analyser.polarity_scores(sent.text)
            else:
                sentiment = self.analyser(sent.text)
            doc.user_span_hooks[i]["sentiment"] = sentiment
        return doc



if __name__ == "__main__":

    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe("sentiment_analysis")

    # Process a doc and see the results
    doc = nlp("Australian workers are exhausted, unwell, at risk of quitting, and largely unprepared for future workplace challenges driven by automation and artificial intelligence, a new report from the University of Melbourne Work Futures Hallmark Research Initiative reveals. A comprehensive survey of 1,400 Australian workers fielded in June 2022 asked about their experiences at work since the onset of the COVID-19 pandemic. The findings, published in the 2023 State of the Future of Work Report, reveals Australian workers were in poorer physical and mental health since the pandemic began, with prime aged workers (between 25-55 years of age) significantly impacted, one third of whom had considered quitting.")
