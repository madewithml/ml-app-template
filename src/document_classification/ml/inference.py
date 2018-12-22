import os
import torch

class Inference(object):
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def predict_y(self, X):
        # Vectorize
        vectorized_X = self.vectorizer.vectorize(X)
        vectorized_X = torch.tensor(vectorized_X).unsqueeze(0)

        # Forward pass
        self.model.eval()
        y_pred = self.model(x_in=vectorized_X, apply_softmax=True)

        # Top y
        y_prob, indices = y_pred.max(dim=1)
        index = indices.item()

        # Predicted y
        y = vectorizer.y_vocab.lookup_index(index)
        probability = y_prob.item()
        return {'y': y, 'probability': probability}

    def predict_top_k(self, X, k):
        # Vectorize
        vectorized_X = self.vectorizer.vectorize(X)
        vectorized_X = torch.tensor(vectorized_X).unsqueeze(0)

        # Forward pass
        self.model.eval()
        y_pred = self.model(x_in=vectorized_X, apply_softmax=True)

        # Top k categories
        y_prob, indices = torch.topk(y_pred, k=k)
        probabilities = y_prob.detach().numpy()[0]
        indices = indices.detach().numpy()[0]

        # Results
        results = []
        for probability, index in zip(probabilities, indices):
            y = self.vectorizer.y_vocab.lookup_index(index)
            results.append({'y': y, 'probability': probability})

        return results