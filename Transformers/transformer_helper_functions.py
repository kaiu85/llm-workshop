from collections import OrderedDict
from math import ceil
import torch
import wikipedia as wiki

def chunkify_and_get_answers(frage, seiteninhalt, tokenizer, model):
    
    inputs = tokenizer.encode_plus(frage, # Frage
                                     seiteninhalt, # Gesamter Artikeltext
                                     add_special_tokens=True, # Fügt am Anfang ein CLS-Token
                                     # und zwischen Frage und Artikeltext, sowie am Ende, ein
                                     # SEP-Token ein
                                     return_tensors="pt", # Gibt die Token-Liste im
                                     # PyTorch-Format-zurück
                                     padding=False # WICHTIG: Wir füllen die Token-Liste
                                     # hier **NICHT** mit PAD-Tokens bis zur Maximallänge
                                     # auf, da wir GENAU WISSEN WOLLEN, WIE LANGE
                                     # der gesamte Input ist
                                    )


    input_ids = inputs["input_ids"].tolist()[0]

    frage_maske = inputs['token_type_ids'].lt(1)

    frage_input_ids = torch.masked_select(inputs['input_ids'], frage_maske)
    frage_token_type_ids = torch.masked_select(inputs['token_type_ids'], frage_maske)
    frage_attention_mask = torch.masked_select(inputs['attention_mask'], frage_maske)


    text_input_ids = torch.masked_select(inputs['input_ids'], ~frage_maske)[:-1]
    text_token_type_ids = torch.masked_select(inputs['token_type_ids'], ~frage_maske)[:-1]
    text_attention_mask = torch.masked_select(inputs['attention_mask'], ~frage_maske)[:-1]

    max_len = model.config.max_position_embeddings
    chunk_size = max_len - frage_maske.sum() - 1
    n_chunks = int(ceil(len(text_input_ids) / chunk_size))

    chunked_inputs = []

    answers = []

    for i in range(n_chunks):

        start_index = i*chunk_size        
        end_index = i*chunk_size + chunk_size

        if end_index > len(text_input_ids) - 1:
            end_index = len(text_input_ids) - 1            

        chunk = {

            'input_ids': torch.cat((frage_input_ids, text_input_ids[start_index:end_index], torch.tensor([103]))).unsqueeze(dim=0),
            'token_type_ids': torch.cat((frage_token_type_ids, text_token_type_ids[start_index:end_index], torch.tensor([1]))).unsqueeze(dim=0),
            'attention_mask': torch.cat((frage_attention_mask, text_attention_mask[start_index:end_index], torch.tensor([1]))).unsqueeze(dim=0)     

        }

        chunked_inputs.append(chunk)   

        output = model(**chunk)

        start_index = torch.argmax(output['start_logits'])
        end_index = torch.argmax(output['end_logits'])    

        chunk_ids = chunk["input_ids"].tolist()[0] # Token-ID-Liste des gerade verarbeiteten Chunks
        antwort = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(chunk_ids[start_index:end_index+1]))

        answers.append(antwort)
        
    return answers

def get_wikipedia_articles(frage, n = 3):

    wiki.set_lang('de')
    
    suchergebnisse = wiki.search(frage, results = n)
    
    titles = []
    texts = []
    
    for hit in suchergebnisse:
        
        ganze_seite = wiki.page(hit)
        
        titles.append(ganze_seite.title)
        texts.append(ganze_seite.content)
        
    return titles, texts


    
from whoosh.qparser import QueryParser, OrGroup, FuzzyTermPlugin, WildcardPlugin
from whoosh.scoring import BM25F
from whoosh.index import open_dir, create_in
from whoosh.query import FuzzyTerm
from whoosh.fields import Schema, TEXT, ID
import sys
from whoosh.analysis import StopFilter, RegexTokenizer, LowercaseFilter, StemFilter, LanguageAnalyzer
import os

# Die Funktion erhält eine Frage und optional die gewünschte Anzahl der 
# herauszusuchenden Artikel und den Pfad zu dem Verzeichnis, indem 
# der Suchindex gespeichert ist
def get_flexikon_articles(frage, n = 3, index_dir = './data/indexdir'):
    
    # Der Suchindex wird geöffnet
    ix = open_dir(index_dir)
        
    # Der "Parser" wird dazu benutzt, um die Frage
    # entsprechend aufzubereiten
    # Hier wird zunächst ein parser-Object erzeugt und die
    # entsprechenden Optionen angegeben
    parser = QueryParser("content", schema = ix.schema, group = OrGroup)
    parser.add_plugin(FuzzyTermPlugin())
    parser.remove_plugin_class(WildcardPlugin)

    # Wir benutzen den BM25F-Suchalgorithmus
    with ix.searcher(weighting=BM25F) as searcher:
        
        # Hier wird der Zeichenstring "frage" durch den
        # Parser in eine entsprechende "Query" (Anfrage)
        # umgewandelt, die der Suchalgorithmus
        # (searcher) verarbeiten kann
        query = parser.parse(frage)

        # Dem searcher wird die query übergeben, zusammen mit
        # dem Parameter, der bestimmt, wie viele Ergebnisse
        # gefunden werden sollen.
        # Der Searcher gibt dann eine Liste der entsprechenden
        # Suchresultate zurück
        results = searcher.search(query,limit=n,terms=True)

        titles = []
        texts = []

        # Wir iterieren über die Suchresultate und
        # lesen die entsprechenden Titel und
        # Texte aus, und speichern sie in den
        # entsprechenden Listen
        for hit in results:
            titles.append( hit['title'] )
            texts.append( hit['textdata'] )

    # Wir geben die Listen mit den Titeln und
    # Texten der gefundenen Artikel zurück.
    return titles, texts

def create_flexikon_index(data_dir, index_dir = './data/indexdir'):
   

    # Hier müssen wir den Ordner angeben, in dem die einzelnen
    # Flexikon-Textdateien (eine Textdatei enthält einen Flexikon-
    # artikel und der Dateiname entspricht dem Titel des Artikels)
    # zu finden sind. Diesen Ordner erhalten wir als Parameter
    # 'data_dir'
    root = data_dir

    # Das Analysemodul, dass die einzelnen Flexikon-Textdateien verarbeiten wird,
    # um einen suchbaren Index zu erstellen.
    # Da es sich um deutsche Artikel handeln wird, spezifizieren wir "de"(utsch)
    analyzer = LanguageAnalyzer("de")

    # Das schema, nach dem die Artikel katalogisiert werden sollen.
    # Wir wollen uns den Titel (title), den Dateipfad (path), den suchbaren Inhalt (content)
    # und den Originaltext (textdata) für jede einzelne Textdatei merken.
    # Hierbei benutzen wir, dass wir wissen, dass jede einzelne Textdatei genau
    # einem Flexikion-Artikel entspricht, und der Dateiname den Titel des
    # Artikels darstellt.
    schema = Schema(title=TEXT(stored=True, lang = 'de'),path=ID(stored=True),\
                  content=TEXT(analyzer=analyzer, lang = 'de'),textdata=TEXT(stored=True, lang = 'de'))

    # Falls noch kein Verzeichnis für den Index erstellt wurde, 
    # erstellen wir es.
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    # Der Indexwriter "writer" verarbeitet nacheinander die einzelnen Textdateien
    # und schreibt die entsprechende Information in den Index
    ix = create_in(index_dir,schema)
    writer = ix.writer()

    # Wir lassen uns eine Liste von allen Textdateien im Ordner
    # ./data/Flexikon ausgeben. Jede Textdatei enthält einen Flexikon
    # Artikel
    filepaths = [os.path.join(root,i) for i in os.listdir(root)]

    # Wir geben einmal aus, wie viele Dateien wir gefunden haben
    print('Found %d files/articles!' % len(filepaths))

    # Wir iterieren über alle Textdateien
    for path in filepaths:

        # Wir öffnen und lesen die Textdatei
        fp = open(path,'r')
        text = fp.read()

        # Wir geben aus, welche Datei wir gerade verarbeiten
        print('Processing File: ' + path)

        # Wir erzeugen den Titel des Eintrags, in dem wir 
        # vom Dateinamen die letzten vier Zeichen (".txt")
        # abschneiden.
        title = path.split("/")[-1][:-4]

        # Wir geben den Titel aus, unter dem der Artikel gespeichert
        # werden wird
        print('Title: ' + title)

        # Wir geben die Länge der Textdatei in 
        # Buchstaben aus
        print('Length: ' + str(len(text)) + " characters")

        # Wir fügen dem Index die Datei als Eintrag hinzu
        writer.add_document(title=title, path=path,
                            content=text,textdata=text)

        # Wir schließen die Textdatei
        fp.close()

    # Wir speichern die Änderungen am Index
    writer.commit()