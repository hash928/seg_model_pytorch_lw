import numpy as np
from lets_plot import *

# %%

np.random.seed(12)
data = dict(
    cond=np.repeat(['A', 'B'], 200),
    rating=np.concatenate((np.random.normal(0, 1, 200),
                           np.random.normal(1, 1.5, 200)))
)

p = (
    ggplot(data, aes(x='rating', fill='cond'))
    + ggsize(500, 250)
    + geom_density(color='dark_green', alpha=.7)
    + scale_fill_brewer(type='seq')
    + labs(x='Rating', y='Density', fill='Condition')
    + theme(legend_position='right')
)

p.show()