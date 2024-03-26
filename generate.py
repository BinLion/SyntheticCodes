from typer import Typer

from falcon.TextGenerationInference import TGI, GenerateParameters, GenerateRequest

app = Typer

@app.command
def generate(
    endpoint: str,
    region: str = "us-west-2"
):
    model = TGI(endpoint, region)
    stop_words = ["\nclass", "\ndef", "\n#", "\n```"]
    params = GenerateParameters( max_new_tokens=512, 
                            temperature=1, 
                            stop =stop_words, 
                            top_p = 0.95,
                            #return_log_probs = True,
                            )
    req = GenerateRequest("def advantages_equal(musician_1, musician_2): \"\"\"This exercise will determine the intersection of advantages between two musicians and check if they are equal.\"\"\"", params)
    _outputs = model.sm_query(req)
    print(_outputs)
    