import torch
from vllm.utils import direct_register_custom_op

def dump_info_impl(desc:str, input:torch.Tensor) -> None:
    print(desc)
    print(input)

def dump_info_fake(desc:str, input:torch.Tensor) -> None:
    pass

direct_register_custom_op(op_name="dump_info",
                            op_func=dump_info_impl,
                            fake_impl=dump_info_fake,
                            tags=())

def dump_info_op(desc:str, input:torch.Tensor):
    return torch.ops.vllm.dump_info(desc, input)