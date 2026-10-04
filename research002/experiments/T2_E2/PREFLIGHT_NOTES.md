# Implementation preflight

Before any research dataset was generated, a small synthetic fixture exposed an incorrect verification assertion about scikit-learn's `n_iter_` attribute: this version resets that counter within each `partial_fit` call. The estimator had received the intended epochs, but `assert model.n_iter_ == 80` failed.

The assertion was corrected to `model.t_ == 80 * len(y)`, which checks the cumulative number of training examples processed. The 80-call training loop, optimizer, scientific protocol, and selection rules were unchanged. No calibration or empirical panel had been run, and none was repeated because of this correction.
