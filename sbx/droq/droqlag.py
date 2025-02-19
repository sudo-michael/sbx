from typing import Any, Dict, Optional, Tuple, Type, Union

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from sbx.tqc.policies import TQCLagPolicy
from sbx.tqc.tqclag import TQCLag
from sbx.buffers.buffers import SafeReplayBuffer

class DroQLag(TQCLag):
    policy_aliases: Dict[str, Type[TQCLagPolicy]] = {
        "MlpPolicy": TQCLagPolicy,
    }

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        qf_learning_rate: Optional[float] = None,
        qfc_learning_rate: Optional[float] = None,
        buffer_size: int = 500_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 2,
        # policy_delay = gradient_steps to follow original implementation
        policy_delay: int = 2,
        top_quantiles_to_drop_per_net: int = 2,
        dropout_rate: float = 0.01,
        layer_norm: bool = True,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[SafeReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        lag_coef: Union[str, float] = "auto",
        cost_limit: float = 25.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            qfc_learning_rate=qfc_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_delay=policy_delay,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            top_quantiles_to_drop_per_net=top_quantiles_to_drop_per_net,
            ent_coef=ent_coef,
            lag_coef=lag_coef,
            cost_limit=cost_limit,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            _init_setup_model=False,
        )

        self.policy_kwargs["dropout_rate"] = dropout_rate
        self.policy_kwargs["layer_norm"] = layer_norm

        if _init_setup_model:
            self._setup_model()
