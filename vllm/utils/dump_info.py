import torch
from vllm.utils import direct_register_custom_op

dump_infos = {}

def dump_info_impl(desc:str, input:torch.Tensor) -> None:
    dump_infos[desc] = input

def dump_info_fake(desc:str, input:torch.Tensor) -> None:
    pass

direct_register_custom_op(op_name="dump_info",
                            op_func=dump_info_impl,
                            fake_impl=dump_info_fake,
                            tags=())

def dump_info_op(desc:str, input:torch.Tensor):
    return torch.ops.vllm.dump_info(desc, input)

def dump_info_print():
    for desc, tensor in dump_infos.items():
        print(f"{desc} : {tensor}")