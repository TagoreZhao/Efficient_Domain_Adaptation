class WrappedGPT:
    """
    Wrap a Linear layer to accumulate per-input-dimension squared L2 norms (row-scaler),
    used by Wanda: |W| * sqrt(E[||x||^2]) per input channel.
    """
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns,), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        # inp: either [bsz, in_features] or [bsz, seq, in_features] depending on where hook is placed
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)  # [1, bsz, in_features]
        tmp = inp.shape[0]

        if isinstance(self.layer, nn.Linear):
            if inp.dim() == 3:
                inp = inp.reshape((-1, inp.shape[-1]))  # [bsz*seq, in_features] (or [bsz, in_features])
            inp = inp.t()  # [in_features, n_tokens]

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.to(torch.float32)
        self.scaler_row += (torch.norm(inp, p=2, dim=1) ** 2) / self.nsamples