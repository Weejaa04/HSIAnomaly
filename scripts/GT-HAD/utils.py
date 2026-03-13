SEED_DICT = {
    'Almond':      42,
    'Pistachio':   42,
    'GarlicStems': 42,
}


def get_params(net):
    """Returns all parameters to optimize."""
    return list(net.parameters())


def img2mask(img):
    """Collapse (1, B, H, W) residual map → normalised (H, W) numpy array."""
    img = img[0].sum(0)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.detach().cpu().numpy()