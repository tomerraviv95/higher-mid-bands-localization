from sklearn.cluster import KMeans


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]

    return merged_list


PROXIMITY_THRESH = 20


def filter_peaks(tuples, L_hat):
    kmeans = KMeans(n_clusters=L_hat).fit(tuples)
    peaks = kmeans.cluster_centers_.astype(int)
    return peaks
