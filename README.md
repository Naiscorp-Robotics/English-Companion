# English-Companion
This project targets in building a comprehensive NLP system for English-Companion robot. The core idea is to create an LLM-based system that can be integrated into a robot (conversation) that can response in bilingual answers accompany with english error corrections. 
eg: Error classification.
    " I want to go to California with my friend" --> true
    " There is many animals in the ground" --> vocab_error 
    " How many cows there are" --> gram_error
eg: Error words detection.
    " there is many species in the ground" --> "is", "species", "in"
    " my appear is quite good" --> "appear"