from einops import einsum
from jaxtyping import Float, Bool, Int
import math
import torch
from torch import device, dtype, Tensor, sigmoid


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

    def forward(
        self, x: Float[Tensor, "... d_in"]
    ) -> Float[Tensor, "... d_out"]:
        # 2 * batch * sequence * d_in * d_out FLOPs
        return einsum(
            x,
            self.weight,
            "batch sequence d_in, d_out d_in -> batch sequence d_out",
        )
        # return x @ self.weight.T


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(
                num_embeddings, embedding_dim, device=device, dtype=dtype
            )
        )
        torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )
        torch.nn.init.ones_(self.weight)
        pass

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        output = x_norm * self.weight
        return output.to(in_dtype)


class SiLU(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = int(round(8 / 3) * d_model / 64) * 64
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.silu = SiLU()

    def forward(
        self, x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RoPE(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        pos = torch.arange(max_seq_len, device=device)
        theta_ik = pos.outer(
            1.0
            / (
                theta
                ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k)
            )
        )
        self.register_buffer("cos_cache", theta_ik.cos(), persistent=False)
        self.register_buffer("sin_cache", theta_ik.sin(), persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x0 = x[..., ::2]
        x1 = x[..., 1::2]
        return torch.stack((-x1, x0), dim=-1).flatten(-2)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        sin = self.sin_cache[token_positions].to(x.device)
        cos = self.cos_cache[token_positions].to(x.device)
        xdtype = x.dtype
        x = x.float()
        sin = torch.repeat_interleave(sin, 2, -1)
        cos = torch.repeat_interleave(cos, 2, -1)
        res = x * cos + self._rotate_half(x) * sin
        return res.to(xdtype)


# softmax(v,i) = exp(v_i) / sum(exp(v_j),j=1...n)
def softmax(in_features: Tensor, dim: int = -1) -> Tensor:
    # the exp must get a large value that cause overflow(be inf)
    # cause that inf/inf = NaN
    # notice that softmax is shift invariance, so we can add a shift on input
    # we chose the max value of input
    # make the softmax output values in [0,1)
    max_val = in_features.max(dim=dim, keepdim=True).values
    shifted = in_features - max_val
    exp_shifted = shifted.exp()
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)


def attention(
    query: Float[Tensor, " ... queries d_k"],
    key: Float[Tensor, " ... keys d_k"],
    value: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = query.shape[-1]
    attn_out = einsum(
        query, key, " ... queries d_k, ... keys d_k -> ... queries keys"
    ) / math.sqrt(d_k)
    if mask is not None:
        attn_out = attn_out.masked_fill(mask == 0, float("-inf"))
    # dim = -1 because we need to do softmax on keys dim
    # so that every query has a sum of weight 1 on all of keys
    attn_weight = softmax(attn_out)
    out = einsum(
        attn_weight,
        value,
        "... queries keys, ... keys d_v -> ... queries d_v",
    )
    return out


def attn_mask(Q: Tensor, K: Tensor):
    # Q: [B,H,t_q,D]
    # K: [B,H,t_k,D]
    # we should get a upper triangle, with shape
    # [t_q,t_k] that mask the future token
    # when we compute the next token
    t_q = Q.size(-2)
    t_k = K.size(-2)

    # there are three points to be point out:
    #
    # 1. The True means unmask, visible for attention,
    # so we get the reverse result
    #
    # 2. Diagonal=1 will exclude the main diagonal
    #
    # 3. In the attention function, we mask on the answer of pre-softmax proj
    # which is a Linear transformation of Q @ K with query and key dims which
    # are meaningful and necessary for mask.
    # the batch and head dim we don't need to mask it
    # so we insert two cast dimension above the answer
    return ~torch.triu(
        torch.ones(t_q, t_k, dtype=torch.bool, device=Q.device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)


class MutiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.k_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.v_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.output_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.rope: RoPE | None = None
        self.token_position: Tensor | None = token_positions
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device, dtype)

    # TODO: merge Q,K,V to a single weight proj, so that we can reduce
    # compute amount
    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape

        def _split_head(_x: Tensor) -> Tensor:
            return _x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # reverse of _split_head
        def _merge_head(_x: Tensor) -> Tensor:
            return _x.transpose(1, 2).contiguous().view(B, T, self.d_model)

        Q = _split_head(self.q_proj.forward(x))
        K = _split_head(self.k_proj.forward(x))
        V = _split_head(self.v_proj.forward(x))
        # apply RoPE on Q and K if we can
        if self.rope is not None:
            token_positions = self.token_position
            if token_positions is None:
                token_positions = torch.arange(T, device=x.device)
            else:
                token_positions = token_positions.to(x.device)
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
        attn_out = attention(Q, K, V, attn_mask(Q, K))
        # apply to output Linear transformation
        return self.output_proj(_merge_head(attn_out))


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MutiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.attn(self.ln1(x)) + x
        y: Tensor = h + self.ffn(self.ln2(h))
        return y


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        if max_seq_len is None:
            max_seq_len = context_length

        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        y = self.token_embeddings(x)
        for layer in self.layers:
            y = layer(y)
        y = self.ln_final(y)
        y = self.lm_head(y)
        return y
