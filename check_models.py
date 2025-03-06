import google.generativeai as genai

genai.configure(api_key="AIzaSyDWDBrb_lSbHiaUyLV4si643hQCIo8Ud-Q")
models = genai.list_models()
for model in models:
    print(model.name)
