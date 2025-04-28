
class BehaviorRecognizer:
    """
    Compares processed keypoint readings (already projected into D-dimensional space)
    against a candidate template (of shape (window_size, D)) using a sliding window.
    The per-dimension error is computed as the maximum absolute difference between the
    window's values and the candidate template.
    Returns:
      scalar_diff: overall maximum error (scalar),
      per_dim_diff: vector of maximum errors per dimension,
      data_arr: the buffered (processed) window.
    """
    def __init__(self, template, D):
         # candidate_template is provided from the configuration loader.
        self.template = template   # shape: (window_size, D)
        self.template_length = self.template.shape[0]
        self.D = D
        self.buffer = []

    def update(self, keypoint_reading):

        self.buffer.append(keypoint_reading.keypoints)

        # If we don't have enough samples yet, return None.
        if len(self.buffer) < self.template_length:
            return None
        
        # Keep only the most recent samples matching the template length.
        if len(self.buffer) > self.template_length:
            self.buffer = self.buffer[-self.template_length:]

        # Stack
        data_arr = np.array(self.buffer)  # shape: (template_length, n_features)

        # Compute similarity: max absolute difference
        diff = np.abs(data_arr - self.template)
        per_dim_diff = np.max(diff, axis=0)
        scalar_diff = np.max(per_dim_diff)
         
        return scalar_diff, per_dim_diff, data_arr

def triangulate(keypoints_2D):




def ProcessBehavior(behavior_params, BehaviorQueue, stop_event):

    while not stop_event.is_set():
        keypoints_2D = BehaviorQueue.get()

