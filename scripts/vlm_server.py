# 中文注释: 导入 socket 标准库，用 TCP socket 给评测脚本提供一个本地 VLM 推理服务。
import socket
# 中文注释: 导入 PyTorch；这里用于模型推理、半精度张量转换和禁用初始化函数。
import torch
# 中文注释: 导入 json；客户端和服务端通过 JSON 传输图像列表、语言指令和文本响应。
import json
# 中文注释: 导入 argparse；脚本直接运行时用它解析 host、port、模型路径等命令行参数。
import argparse
# 中文注释: 导入 os；后面会拼接模型目录里的 tokenizer 子路径。
import os
# 中文注释: 导入 time；用于统计一次 VLM 生成动作文本花费的时间。
import time
# 中文注释: 从 tqdm 导入进度条，用来在加载 checkpoint 时显示进度状态。
from tqdm import tqdm
# 中文注释: 导入 base64；客户端把图片编码成 base64 字符串后通过 JSON 发送。
import base64
# 中文注释: 从 io 导入 BytesIO；把 base64 解码后的字节包装成类文件对象给 PIL 读取。
from io import BytesIO
# 中文注释: 从 PIL 导入 Image；用于把客户端发来的图片字节转换成 RGB 图像。
from PIL import Image
# 中文注释: 导入 re 正则库；当前文件暂未使用，保留可能是为了后续解析 VLM 文本动作。
import re

# 中文注释: 导入 HuggingFace tokenizer 自动加载器和配置加载器，用于读取 NaVILA/LLaVA 模型配置。
from transformers import AutoTokenizer, AutoConfig
# 中文注释: 导入 LLaVA/NaVILA 的图像处理、prompt token 化、停止词和模型名解析工具。
from llava.mm_utils import KeywordsStoppingCriteria, process_image, tokenizer_image_token, get_model_name_from_path
# 中文注释: 导入图像占位 token 的索引，tokenizer_image_token 会用它把 <image> 替换成模型内部图像 token。
from llava.constants import IMAGE_TOKEN_INDEX
# 中文注释: 导入对话模板和分隔符类型，用来按指定 conv_mode 构造模型期望的聊天 prompt。
from llava.conversation import SeparatorStyle, conv_templates
# 中文注释: 导入 NaVILA/LLaVA 的模型加载函数，用 checkpoint 路径创建 tokenizer、模型和 image_processor。
from llava.model.builder import load_pretrained_model
from safe_vln.actions import action_from_id
from safe_vln.model import SafeNavilaActorCritic


# 中文注释: 定义 VLMServer 类；它负责加载 NaVILA 模型，并通过 TCP 接收图像+指令后返回动作文本。
class VLMServer:
    # 中文注释: 这个 docstring 概括类用途：给 IsaacLab 评测流程提供轻量 TCP 封装的 NaVILA 推理服务。
    """Small TCP server that wraps NaVILA inference for IsaacLab evaluation."""

    # 中文注释: 初始化服务对象；传入的 args 来自命令行，里面包含模型路径、设备、端口等运行配置。
    def __init__(self, args):
        # 中文注释: 保存命令行参数，后续加载模型、选择设备、构造 prompt 和启动 socket 都会用到。
        self.args = args
        # 中文注释: 先把 tokenizer 置空；setup 阶段加载 checkpoint 后会写入真正的 tokenizer。
        self.tokenizer = None
        # 中文注释: 先把模型对象置空；setup 阶段会加载 NaVILA/LLaVA 模型到这里。
        self.model = None
        # 中文注释: 先把图像预处理器置空；加载模型后会保存与该模型匹配的 image_processor。
        self.image_processor = None
        # 中文注释: 预留视觉塔引用；当前逻辑没有直接使用，但保留这个字段方便扩展或调试模型结构。
        self.vision_tower = None
        self.safe_model = None
        self.safe_policy_version = None
        # 中文注释: 立即执行模型加载和初始化，让服务开始监听前就准备好推理所需资源。
        self.setup()
        if getattr(args, "safe_checkpoint", None):
            self._load_safe_checkpoint(args.safe_checkpoint)

    def _load_safe_checkpoint(self, checkpoint_path):
        """Load a LoRA adapter and the two Safe-VLN critic heads."""
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load a Safe-VLN checkpoint") from exc
        adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        self.safe_model = SafeNavilaActorCritic(self.model, self.tokenizer).to(self.args.device)
        self.safe_model.load_safe_heads(checkpoint_path, map_location=self.args.device)
        self.safe_model.eval()
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r", encoding="utf-8") as state_file:
                trainer_state = json.load(state_file)
            self.safe_policy_version = int(
                trainer_state.get("policy_version", 0)
            )
        else:
            self.safe_policy_version = 0

    # 中文注释: 定义 setup 方法；它完成一次性初始化，避免每个请求都重复加载大模型。
    def setup(self):
        # 中文注释: 这个 docstring 说明 setup 的目标：只加载一次 tokenizer/model，并在所有 episode 间复用。
        """Load tokenizer/model once, then keep them resident for all episodes."""
        # 中文注释: 禁用默认参数初始化，因为后面会直接加载 checkpoint 权重，随机初始化没有意义且浪费显存/时间。
        self._disable_initializers()
        # 中文注释: 先初始化 tokenizer 和模型配置，为后续 checkpoint 加载准备模型元信息。
        self._initialize_tokenizer_and_model()
        
        # 中文注释: 只支持 W16A16 精度路径；如果命令行选择该精度，就按 fp16/全精度 checkpoint 加载。
        if self.args.precision == "W16A16":
            # 中文注释: 加载 NaVILA checkpoint，并把 tokenizer、模型和图像预处理器保存到当前 server。
            self._load_checkpoint_w16a16()
        # 中文注释: 如果传入其他精度字符串，当前脚本没有对应加载逻辑，需要直接报错提醒用户。
        else:
            # 中文注释: 抛出不支持精度的异常，避免服务用错误模型配置继续运行。
            raise ValueError(f"Precision {self.args.precision} not supported")

    # 中文注释: 定义禁用初始化函数的方法；它通过 monkey patch 跳过 Linear、LayerNorm 和 init API 的默认初始化。
    def _disable_initializers(self):
        # Loading checkpoint weights does not need random parameter
        # initialization. Disabling it saves startup time and avoids a memory
        # spike while the large VLM is being constructed.
        # 中文注释: 把 Linear.reset_parameters 替换成空函数，避免创建 Linear 层时随机初始化权重。
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        # 中文注释: 把 LayerNorm.reset_parameters 替换成空函数，避免 LayerNorm 创建时额外初始化参数。
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        # 中文注释: 禁用 kaiming_uniform_ 初始化函数；加载 checkpoint 时不需要重新采样这些权重。
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        # 中文注释: 禁用 kaiming_normal_ 初始化函数；减少大模型构建阶段的无用计算。
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None
        # 中文注释: 禁用 uniform_ 初始化函数；避免 checkpoint 覆盖前产生临时随机权重。
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        # 中文注释: 禁用 normal_ 初始化函数；同样是为了节省启动时间和内存峰值。
        torch.nn.init.normal_ = lambda *args, **kwargs: None

    # 中文注释: 定义 tokenizer 和配置初始化方法；这里先读取 tokenizer 与模型 config，后面再真正加载权重。
    def _initialize_tokenizer_and_model(self):
        # The checkpoint stores the tokenizer under the llm/ subdirectory while
        # the full multimodal config lives at model_path.
        # 中文注释: 从 model_path/llm 加载语言模型 tokenizer；NaVILA checkpoint 把 tokenizer 放在 llm 子目录下。
        self.tokenizer = AutoTokenizer.from_pretrained(
            # 中文注释: 拼出 tokenizer 所在目录，并关闭 fast tokenizer 以匹配该模型仓库的 tokenizer 实现。
            os.path.join(self.args.model_path, "llm"), use_fast=False
        # 中文注释: 结束 AutoTokenizer.from_pretrained 调用，把结果赋给 self.tokenizer。
        )
        # 中文注释: 读取多模态模型配置；trust_remote_code=True 允许加载模型仓库自定义的配置类。
        config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)

    # 中文注释: 定义 W16A16 checkpoint 加载逻辑；该路径加载论文/评测使用的 NaVILA 模型。
    def _load_checkpoint_w16a16(self):
        # 中文注释: 这个 docstring 标明本函数加载的是 full-precision/fp16 版本的 NaVILA checkpoint。
        """Load the full-precision/fp16 NaVILA checkpoint used by the paper."""
        # 中文注释: 创建长度为 1 的进度条；虽然只有一次加载，但能在终端显示“正在加载 checkpoint”。
        pbar = tqdm(range(1))
        # 中文注释: 设置进度条文字，让用户知道当前耗时步骤是加载 checkpoint shards。
        pbar.set_description("Loading checkpoint shards")
        # 中文注释: 遍历这个单步进度条；循环体里执行真正的模型加载。
        for _ in pbar:
            # self.model.llm = load_checkpoint_and_dispatch(
            #     self.model.llm,
            #     os.path.join(self.args.model_path, "llm"),
            #     no_split_module_classes=[
            #         "OPTDecoderLayer",
            #         "LlamaDecoderLayer",
            #         "BloomBlock",
            #         "MPTBlock",
            #         "DecoderLayer",
            #         "CLIPEncoderLayer",
            #     ],
            # ).to(self.args.device)
            # 中文注释: 根据模型路径解析模型名称；注意这里沿用原代码里的全局 args，而不是 self.args。
            model_name = get_model_name_from_path(args.model_path)
            # 中文注释: 从 checkpoint 路径加载 tokenizer、模型、图像预处理器和上下文长度。
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, None)
            # 中文注释: 用 load_pretrained_model 返回的 tokenizer 覆盖初始化阶段加载的 tokenizer，确保与模型完全一致。
            self.tokenizer =  tokenizer
            # 中文注释: 保存加载好的 NaVILA/LLaVA 模型，后续 process_request 会调用 self.model.generate。
            self.model = model
            # 中文注释: 保存和该模型匹配的图像预处理器，用于把 PIL 图像转成模型输入张量。
            self.image_processor = image_processor
        # 中文注释: 把模型移动到命令行指定设备，通常是 cuda，用于后续 GPU 推理。
        self.model = self.model.to(self.args.device)

    # 中文注释: 定义 TCP 服务启动函数；默认监听 localhost:12345，但主入口会用命令行参数覆盖。
    def start_server(self, host='localhost', port=12345):
        # 中文注释: 这个 docstring 说明通信协议：每个 TCP 连接处理一个带长度头的 JSON 请求。
        """Serve one request per TCP connection using a length-prefixed JSON protocol."""
        # 中文注释: 创建 IPv4/TCP socket，作为 VLM 推理服务的监听 socket。
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 中文注释: 把服务 socket 绑定到指定 host 和 port，让客户端可以连接这个地址。
        server_socket.bind((host, port))
        # 中文注释: 开始监听连接，backlog=1 表示同时排队的连接数很小，符合单请求串行推理服务。
        server_socket.listen(1)
        # 中文注释: 打印监听地址，方便确认服务已经启动并等待客户端请求。
        print(f"VLM Server listening on {host}:{port}")

        # 中文注释: 持续运行服务主循环；每次 accept 一个客户端连接，处理完后关闭该连接。
        while True:
            # 中文注释: 阻塞等待客户端连接；conn 是本次通信 socket，addr 是客户端地址。
            conn, addr = server_socket.accept()
            # 中文注释: 用 try/finally 包住请求处理，确保无论成功还是异常都会关闭 conn。
            try:
                # Protocol: 8-byte big-endian payload length, followed by JSON
                # containing base64 JPEG frames and the natural-language query.
                # 中文注释: 先读取 8 字节长度头；客户端用 big-endian 编码告诉服务端后面 JSON payload 有多长。
                size_data = conn.recv(8)
                # 中文注释: 如果只是端口探测或客户端提前断开，忽略这次空连接并继续服务后续请求。
                if len(size_data) < 8:
                    print(f"Ignoring incomplete request header from {addr}")
                    continue
                # 中文注释: 把 8 字节长度头转成整数，得到接下来需要接收的数据字节数。
                size = int.from_bytes(size_data, 'big')
                
                # Receive the actual data
                # 中文注释: 初始化二进制缓冲区，用来累计接收到的 JSON payload 字节。
                data = b''
                # 中文注释: 只要累计长度还没达到 size，就继续从 socket 读取后续分片。
                while len(data) < size:
                    # 中文注释: 每次最多读取 4096 字节，避免一次 recv 假设能拿到完整 payload。
                    packet = conn.recv(4096)
                    # 中文注释: 如果收到空 packet，说明客户端提前断开，停止继续读取。
                    if not packet:
                        # 中文注释: 跳出接收循环，后面会尝试解析当前已经收到的数据。
                        break
                    # 中文注释: 把本次收到的分片追加到 data 缓冲区中。
                    data += packet

                # 中文注释: 如果客户端在 payload 未发送完整时断开，忽略这次坏请求，避免服务进程退出。
                if len(data) < size:
                    print(f"Ignoring incomplete request payload from {addr}: got {len(data)} of {size} bytes")
                    continue

                # Parse the received data
                # 中文注释: 将接收到的 UTF-8 JSON 字节解码并解析成 Python 字典。
                request = json.loads(data.decode())
                # 中文注释: 取出客户端发送的图像列表；每个元素通常是 base64 编码的 JPEG/PNG 字符串。
                images = request['images']
                # 中文注释: 取出自然语言导航指令，例如“go to the door”这类 VLN 任务描述。
                query = request['query']

                # The response is plain text, for example "move forward 50cm";
                # navila_eval.py later parses this into velocity and duration.
                # 中文注释: 调用 VLM 推理流程，根据多帧图像和语言指令生成下一步动作文本。
                response = self.process_request(images, query)
                
                # Send response back
                # 中文注释: 把动作文本响应编码成 JSON 字节，保持和请求一致的结构化传输方式。
                response_bytes = json.dumps(response).encode()
                # 中文注释: 单独保护发送过程；客户端可能在模型推理期间已经断开连接。
                try:
                    # 中文注释: 先发送 8 字节响应长度头，让客户端知道后面要读取多少字节。
                    conn.sendall(len(response_bytes).to_bytes(8, 'big'))
                    # 中文注释: 再发送真正的 JSON 响应内容。
                    conn.sendall(response_bytes)
                # 中文注释: 如果客户端在发送响应时断开连接，就捕获 BrokenPipeError 并打印提示。
                except BrokenPipeError:
                    # 中文注释: 输出具体客户端地址，方便定位哪个连接提前关闭。
                    print(f"Client {addr} disconnected while sending response")
                # 中文注释: 捕获其他发送异常，避免单个失败请求杀死整个服务进程。
                except Exception as e:
                    # 中文注释: 打印发送响应时的异常信息，便于调试 socket 通信问题。
                    print(f"Error sending response to {addr}: {str(e)}")

            # 中文注释: 捕获单个坏请求的异常，避免 JSON 解析、字段缺失或推理错误直接杀死整个服务。
            except Exception as e:
                # 中文注释: 打印坏请求来源和异常内容，服务端随后继续等待下一次连接。
                print(f"Error processing request from {addr}: {str(e)}")
            # 中文注释: 无论请求解析、推理或响应发送是否成功，最后都要关闭本次连接。
            finally:
                # 中文注释: 关闭当前客户端连接，服务端回到 while True 等待下一次连接。
                conn.close()

    # 中文注释: 定义单次请求处理函数；输入多帧图像和一句导航指令，输出 VLM 生成的动作文本。
    def process_request(self, images, query):
        # 中文注释: 这个 docstring 说明本函数会把历史帧+当前帧和语言指令送入 NaVILA。
        """Run NaVILA on eight sampled history/current frames and one instruction."""
        # Convert base64/PIL inputs to the tensor layout expected by LLaVA/NaVILA.
        # 中文注释: 将客户端传来的图片列表转成 NaVILA 图像编码器需要的张量格式。
        image_tensor = process_images(images, self.image_processor, self.model.config)
        # 中文注释: 把图像张量移动到推理设备并转成 float16，以匹配 W16A16/fp16 模型推理。
        image_tensor = image_tensor.to(self.args.device, dtype=torch.float16)

        # Prompt mirrors the VLA action space used in evaluation: turn, move
        # forward, or stop. The client sends exactly eight frames by default.
        # 中文注释: 复制指定 conv_mode 的对话模板，避免修改全局模板对象。
        conv = conv_templates[self.args.conv_mode].copy()
        # 中文注释: 把客户端传来的 query 作为导航任务指令放进 prompt。
        instruction = query
        # 中文注释: 定义单帧图像占位符；后面会按视频帧数重复插入 prompt。
        image_token = "<image>\n"
        # 中文注释: 构造给 VLM 的完整问题文本，描述机器人角色、历史观察、当前观察和可选动作空间。
        qs = (
            # 中文注释: prompt 第一段告诉模型它扮演的是执行导航任务的机器人。
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            # 中文注释: prompt 第二段插入历史帧图像 token 和当前帧图像 token，并写入具体任务指令。
            f'of historical observations {image_token * (self.args.num_video_frames-1)}, and current observation <image>\n. Your assigned task is: "{instruction}" '
            # 中文注释: prompt 第三段要求模型根据连续图像分析下一步动作，动作可以是转向或前进。
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            # 中文注释: prompt 第四段补充动作也可以是前进指定距离，或者任务完成时输出 stop。
            f"degree, moving forward a certain distance, or stop if the task is completed."
        # 中文注释: 结束多行 prompt 字符串拼接，得到最终用户侧消息文本 qs。
        )
        # 中文注释: 把用户消息追加到对话模板中；conv.roles[0] 通常表示 user/human 角色。
        conv.append_message(conv.roles[0], qs)
        # 中文注释: 追加 assistant 占位消息 None，表示接下来希望模型生成这个角色的回复。
        conv.append_message(conv.roles[1], None)
        # 中文注释: 根据对话模板生成最终 prompt 字符串，里面包含角色名、分隔符和图像 token。
        prompt = conv.get_prompt()

        # Stop when the conversation separator is generated so downstream
        # parsing receives only the assistant's action text.
        # 中文注释: 将含 <image> 的 prompt token 化，并把图像 token 替换为 IMAGE_TOKEN_INDEX 后移到推理设备。
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.args.device)
        # 中文注释: 根据对话模板选择停止字符串；双分隔符模板使用 sep2，否则使用 sep。
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # 中文注释: 把停止字符串放进关键词列表，生成时遇到它就停止输出。
        keywords = [stop_str]
        # 中文注释: 创建 LLaVA 的停止准则对象，让 generate 在输出对话分隔符时提前结束。
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # 中文注释: 进入 inference_mode，关闭梯度和 autograd 记录，降低推理显存和计算开销。
        if self.safe_model is not None:
            with torch.inference_mode():
                safe_output = self.safe_model.act(
                    input_ids,
                    images=[image_tensor],
                    deterministic=getattr(self.args, "safe_deterministic", True),
                )
            action = action_from_id(safe_output["action_id"])
            return {
                "protocol_version": "safe-vln-go2-v1",
                **safe_output,
                "policy_version": self.safe_policy_version,
                "action": action.text,
            }

        with torch.inference_mode():
            # 中文注释: 记录生成开始时间，用于统计本次 VLM 推理耗时。
            start_time = time.time()
            # 中文注释: 调用多模态模型生成动作回复；input_ids 是文本，images 是对应的视频帧张量。
            output_ids = self.model.generate(
                # 中文注释: 传入 prompt token ids，告诉语言模型当前对话上下文和图像占位位置。
                input_ids,
                # 中文注释: 传入图像张量列表；NaVILA/LLaVA generate 会把它交给视觉编码器处理。
                images=[image_tensor],
                # 中文注释: 关闭随机采样，让模型用确定性解码，评测结果更稳定。
                do_sample=False,
                # 中文注释: temperature 设为 0，与 do_sample=False 一起表示不引入采样随机性。
                temperature=0,
                # 中文注释: 不启用 nucleus sampling；确定性解码场景下 top_p 不需要设置。
                top_p=None,
                # 中文注释: beam 数设为 1，使用贪心解码而不是多 beam 搜索。
                num_beams=1,
                # 中文注释: 限制本次最多生成 512 个新 token，避免异常情况下无限长输出。
                max_new_tokens=512,
                # 中文注释: 启用 KV cache，加速自回归生成过程。
                use_cache=True,
                # 中文注释: 传入停止准则列表，遇到对话分隔符时终止生成。
                stopping_criteria=[stopping_criteria],
            # 中文注释: 结束 self.model.generate 调用，output_ids 保存模型生成的 token 序列。
            )
            # 中文注释: 计算本次生成耗时，便于观察 VLM 推理速度。
            generation_time = time.time() - start_time
            # 中文注释: 打印生成耗时，帮助调试服务吞吐和延迟。
            print(f"Model generation took {generation_time:.2f} seconds")
            # print("input_ids:", input_ids)

        # 中文注释: 将生成的 token ids 解码成文本，并去掉特殊 token，得到可供导航解析的动作字符串。
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # 中文注释: 去掉首尾空白后返回模型回复，例如 turn left、move forward 或 stop。
        return outputs.strip()


# 中文注释: 定义图像预处理函数；它把 PIL/base64 图像列表转换成模型 image_processor 期望的张量。
def process_images(images, image_processor, model_cfg):
    # 中文注释: 这个 docstring 说明 images 可以是 PIL Image，也可以是客户端传来的 base64 字符串。
    """Process a list of images (either PIL Images or base64 strings)."""
    # 中文注释: 把当前模型的 image_processor 写入 config，process_image 会从 model_cfg 中读取预处理配置。
    model_cfg.image_processor = image_processor
    # 中文注释: 创建列表，逐帧保存 process_image 处理后的图像张量。
    processed_images = []
    
    # 中文注释: 遍历客户端发来的每一帧图像，逐个解码和预处理。
    for image in images:
        # 中文注释: 如果当前图像是字符串，说明它来自 socket JSON，需要先按 base64 解码成图片。
        if isinstance(image, str):
            # Client-side evaluation sends JPEG frames as base64 strings to keep
            # the socket payload JSON-serializable.
            # 中文注释: 用 try 捕获坏图像、截断 payload 或 base64 解码失败等问题。
            try:
                # Decode base64 string to PIL Image
                # 中文注释: 将 base64 字符串解码成字节，再用 PIL 打开并统一转换为 RGB 三通道图像。
                image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
            # 中文注释: 如果解码失败，捕获异常并走兜底图像逻辑，避免整个服务崩溃。
            except Exception as e:
                # 中文注释: 打印解码失败原因，便于定位客户端传图或网络传输问题。
                print(f"Error decoding base64 image: {e}")
                # Create a blank image if decoding fails
                # 中文注释: 创建一张黑色 RGB 占位图，让本次请求仍然能继续进入模型流程。
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # process_image applies the model-specific resize/crop/normalization.
        # 中文注释: 调用 LLaVA/NaVILA 的单图预处理，执行模型需要的 resize、crop、normalize 等步骤。
        processed_image = process_image(image, model_cfg, None)
        # 中文注释: 把当前帧预处理后的张量加入列表，等所有帧处理完后再尝试堆叠成视频张量。
        processed_images.append(processed_image)

    # 中文注释: 如果所有帧形状一致，就可以堆叠成一个 [T, C, H, W] 风格的 batch/video 张量。
    if all(x.shape == processed_images[0].shape for x in processed_images):
        # 中文注释: 沿第 0 维堆叠所有帧，形成模型一次推理所需的多帧图像输入。
        processed_images = torch.stack(processed_images, dim=0)
    # 中文注释: 返回预处理后的图像张量或图像张量列表，交给 process_request 继续移动设备和送入模型。
    return processed_images


# 中文注释: 脚本入口保护；只有直接运行 vlm_server.py 时才会解析参数并启动服务。
if __name__ == "__main__":
    # 中文注释: 创建命令行解析器，用来读取服务地址、模型路径、精度和推理设备等参数。
    parser = argparse.ArgumentParser()
    # 中文注释: 注册 --host 参数，决定 TCP 服务绑定的主机名或 IP。
    parser.add_argument("--host", type=str, default='localhost', help="Host to bind the server")
    # 中文注释: 注册 --port 参数，决定 TCP 服务监听端口；默认 54321。
    parser.add_argument("--port", type=int, default=54321, help="Port to bind the server")
    # 中文注释: 注册 --model_path 参数，指定 NaVILA/LLaVA checkpoint 根目录，这是必填项。
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    # 中文注释: 注册 --precision 参数，当前代码只支持 W16A16 路径。
    parser.add_argument("--precision", type=str, default="W16A16", help="compute precision")
    # 中文注释: 注册 --conv_mode 参数，选择 LLaVA 对话模板，默认使用 llama_3。
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    # 中文注释: 注册 --device 参数，指定模型推理设备，通常是 cuda。
    parser.add_argument("--device", type=str, default="cuda")
    # 中文注释: 注册 --num_video_frames 参数，控制 prompt 中历史帧+当前帧的图像 token 数量。
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--safe_checkpoint", type=str, default=None)
    parser.add_argument("--safe_deterministic", action=argparse.BooleanOptionalAction, default=True)
    # 中文注释: 解析命令行参数，并保存为全局 args；注意 _load_checkpoint_w16a16 里也直接引用了这个全局变量。
    args = parser.parse_args()
    
    # 中文注释: 创建 VLMServer 实例；构造过程中会加载 tokenizer、模型和图像预处理器。
    server = VLMServer(args)
    # 中文注释: 按命令行指定的 host/port 启动 TCP 服务，开始等待评测客户端连接。
    server.start_server(host=args.host, port=args.port)
