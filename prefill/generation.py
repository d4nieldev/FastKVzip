"""Generation settings shared by evaluation entry points (no model imports)."""

from dataclasses import asdict, dataclass, fields
import math

GENERATION_REVISION = 2
SAMPLING_REVISION = 1


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 1024
    num_generations: int = 1

    def __post_init__(self):
        for name in ('temperature', 'top_p'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if self.temperature < 0:
            raise ValueError('temperature must be >= 0')
        if not 0 < self.top_p <= 1:
            raise ValueError('top_p must be in (0, 1]')
        for name, minimum in (('top_k', 0), ('max_new_tokens', 1), ('num_generations', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not self.do_sample and self.num_generations > 1:
            raise ValueError('num_generations > 1 requires sampling (temperature > 0 and top_k != 1)')

    @property
    def do_sample(self):
        return self.temperature > 0 and self.top_k != 1

    @classmethod
    def from_args(cls, args):
        return cls(**{field.name: getattr(args, field.name, None) if getattr(args, field.name, None) is not None else field.default for field in fields(cls)})

    def as_dict(self):
        return asdict(self)

    def hf_kwargs(self):
        kwargs = {'do_sample': self.do_sample, 'max_new_tokens': self.max_new_tokens}
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        return kwargs


def add_generation_arguments(parser):
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', '--top_p', type=float, default=1.0)
    parser.add_argument('--top-k', '--top_k', type=int, default=0)
    parser.add_argument('--max-new-tokens', '--max_new_tokens', type=int, default=None,
                        help='Token limit (summaries default to 1024; other tasks retain benchmark limits)')
    parser.add_argument('--num-generations', '--num_generations', type=int, default=1)
