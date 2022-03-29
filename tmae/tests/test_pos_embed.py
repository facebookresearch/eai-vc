from numpy.testing import assert_allclose
from util.pos_embed import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed


def test_get_1d_sincos_pos_embed():
    pos_embed_1d = get_1d_sincos_pos_embed(embed_dim=2, size=16, cls_token=False)
    pos_embed_2d = get_2d_sincos_pos_embed(embed_dim=4, grid_size=16, cls_token=False)
    assert_allclose(pos_embed_1d, pos_embed_2d[:16, :2])


def test_get_1d_sincos_pos_embed_with_cls():
    pos_embed_1d = get_1d_sincos_pos_embed(embed_dim=2, size=16, cls_token=True)
    pos_embed_2d = get_2d_sincos_pos_embed(embed_dim=4, grid_size=16, cls_token=True)
    assert_allclose(pos_embed_1d, pos_embed_2d[: 16 + 1, :2])
