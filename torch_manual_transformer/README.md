This code was written using AI and a lot of back and forth with the model.  We use PyTorch and Cuda but rather than use the 
PyTorch built in transformer code we do that manually for educational purposes.  Performance wise it seems to run close to the same as the PyTorch transformer.
This also follows the concepts of "attention is all you need" whitepaper.

memorizeit.py will by default build a decoder-only model and use it to run 100 epocs on data.txt, we save the best model as best_model.pt
generate.py will load a model and allow you to provide a starter prompt for the model to predict from
visualize.py is some code to help us visualize everything

