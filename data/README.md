# Data Directory
This folder contains data for training, evaluation of all experiments.
## Data Collection Campaign
Generate synthetic data for pre-inference LLM tasks, including: error classification, error words detection.
eg: Error classification.
- Grramar Errors: 
    + Verb Tense/Agreement Errors: He go to school yesterday 
    + Article Errors: I saw elephant
    + Preposition Errors: He is good in math
    + Pronoun Errors: Every student must bring their book
    + Word Form Error: She speaks very quick
- Word Choice Errors: 
    + Inappropriate Word Choice / Register: Sup dude, would you kindly provide me with the documents? ("Sup dude" là ngôn ngữ rất thân mật, còn "would you kindly provide me" rất trang trọng. Kết hợp không phù hợp.)
    + Confusing Similar Meaning Words:  She said me a secret (nên dùng told thay vì said)
    + Collocation Errors: Ví dụ: "Strong tea" (trà đặc) chứ không phải "powerful tea".

    " I want to go to California with my friend" --> true
    " There is many animals in the ground" --> vocab_error 
    " How many cows there are" --> gram_error
eg: Error words detection.
    " there is many species in the ground" --> "is", "species", "in"
    " my appear is quite good" --> "appear"
