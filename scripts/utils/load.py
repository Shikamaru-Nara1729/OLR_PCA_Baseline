import numpy as np
from sklearn.decomposition import PCA
from scripts.utils import constants


# TODO: Add Sphinx documentation
def load_raw(data_path=constants.DATA_PATH, labels_path=constants.LABELS_PATH):
    olr, labels = np.load(data_path), np.load(labels_path)
    olr_processed = np.delete(olr, 6, axis=0) # Removing year 1985 (NaNs)
    # labels = np.delete(labels, 6, axis=0)
    olr_x = np.nan_to_num(olr_processed, copy=True, nan=np.nanmean(olr[3, :, :, :])) # Fixing NaNs in 1982
    return olr_x, labels


def load_anomaly():
    olr_x, labels = load_raw()
    olr_anomaly = olr_x - np.mean(olr_x, axis=0)
    return olr_anomaly, labels


def load_pca():
    olr_data, olr_labels = load_raw()
    pca_olr = olr_data.reshape((40*149, 61*231))
    pca = PCA()
    pca_olr = pca.fit_transform(pca_olr)
    return pca_olr, olr_labels


def load_pca_anomaly():
    olr_data, olr_labels = load_anomaly()
    pca_olr = olr_data.reshape((40*149, 61*231))
    pca = PCA()
    pca_olr = pca.fit_transform(pca_olr)
    return pca_olr, olr_labels