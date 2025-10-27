import torch
from vllm.utils import direct_register_custom_op

def dump_info_impl(desc:str, input:torch.Tensor) -> None:
    print(str)
    print(input)

def dump_info_fake(desc:str, input:torch.Tensor) -> None:
    pass

direct_register_custom_op(op_name="dump_info",
                            op_func=dump_info_impl,
                            mutates_args=["o"],
                            fake_impl=dump_info_fake,
                            tags=())
