set -e

function shorten() {
    python shorten_data.py --src "dataset/pretrain_hq.jsonl" --target "dataset/pretrain_hq_mini.jsonl" --lines 5000
    python shorten_data.py --src "dataset/sft_mini_512.jsonl" --target "dataset/sft_mini_512_mini.jsonl" --lines 10000
    python shorten_data.py --src "dataset/dpo.jsonl" --target "dataset/dpo_mini.jsonl" --lines 100
}

# 预训练(Pretrain):
# LLM首先要学习的并非直接与人交流，而是让网络参数中充满知识的墨水，“墨水” 理论上喝的越饱越好，产生大量的对世界的知识积累。 预训练就是让Model先埋头苦学大量基本的知识，例如从Wiki百科、新闻、书籍整理大规模的高质量训练数据。 这个过程是“无监督”的，即人类不需要在过程中做任何“有监督”的校正，而是由模型自己从大量文本中总结规律学习知识点。 模型此阶段目的只有一个：学会词语接龙。例如我们输入“秦始皇”四个字，它可以接龙“是中国的第一位皇帝”。
function pretrain() {

    python train_pretrain.py \
        --epochs 1 \
        --batch_size 8 \
        --device mps \
        --dtype float32 \
        --dim $dim \
        --log_interval 50 \
        --save_interval 50 \
        --data_path ./dataset/pretrain_hq_mini.jsonl

    # python train_pretrain_mps.py

}

# 有监督微调(Supervised Fine-Tuning):
# 经过预训练，LLM此时已经掌握了大量知识，然而此时它只会无脑地词语接龙，还不会与人聊天。 SFT阶段就需要把半成品LLM施加一个自定义的聊天模板进行微调。 例如模型遇到这样的模板【问题->回答，问题->回答】后不再无脑接龙，而是意识到这是一段完整的对话结束。 称这个过程为指令微调，就如同让已经学富五车的「牛顿」先生适应21世纪智能手机的聊天习惯，学习屏幕左侧是对方消息，右侧是本人消息这个规律。 在训练时，MiniMind的指令和回答长度被截断在512，是为了节省显存空间。就像我们学习时，会先从短的文章开始，当学会写作200字作文后，800字文章也可以手到擒来。 在需要长度拓展时，只需要准备少量的2k/4k/8k长度对话数据进行进一步微调即可（此时最好配合RoPE-NTK的基准差值）。
function sft() {
    python train_full_sft.py \
        --epochs 1 \
        --batch_size 8 \
        --device mps \
        --dtype float32 \
        --dim $dim \
        --log_interval 50 \
        --save_interval 50 \
        --data_path ./dataset/sft_mini_512_mini.jsonl
}

# 人类反馈强化学习(Reinforcement Learning from Human Feedback, RLHF)
# 在前面的训练步骤中，模型已经具备了基本的对话能力，但是这样的能力完全基于单词接龙，缺少正反样例的激励。 模型此时尚未知什么回答是好的，什么是差的。我们希望它能够更符合人的偏好，降低让人类不满意答案的产生概率。 这个过程就像是让模型参加新的培训，从优秀员工的作为例子，消极员工作为反例，学习如何更好地回复。 此处使用的是RLHF系列之-直接偏好优化(Direct Preference Optimization, DPO)。 与PPO(Proximal Policy Optimization)这种需要奖励模型、价值模型的RL算法不同； DPO通过推导PPO奖励模型的显式解，把在线奖励模型换成离线数据，Ref模型输出可以提前保存。 DPO性能几乎不变，只用跑 actor_model 和 ref_model 两个模型，大大节省显存开销和增加训练稳定性。
# 注：RLHF训练步骤并非必须，此步骤难以提升模型“智力”而通常仅用于提升模型的“礼貌”，有利（符合偏好、减少有害内容）也有弊（样本收集昂贵、反馈偏差、多样性损失）。
function dpo() {
    python train_dpo.py \
        --epochs 2 \
        --batch_size 8 \
        --device cpu \
        --dtype float32 \
        --dim $dim \
        --log_interval 50 \
        --save_interval 50 \
        --data_path ./dataset/dpo_mini.jsonl
}

function eval() {
    python eval_model.py \
        --device mps \
        --dim $dim \
        --model_mode 1 # 默认为0：测试pretrain模型效果，设置为1：测试full_sft模型效果
}

# LoRA (Low-Rank Adaptation)
# LoRA是一种高效的参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法，旨在通过低秩分解的方式对预训练模型进行微调。 相比于全参数微调（Full Fine-Tuning），LoRA 只需要更新少量的参数。 LoRA 的核心思想是：在模型的权重矩阵中引入低秩分解，仅对低秩部分进行更新，而保持原始预训练权重不变。 代码可见./model/model_lora.py和train_lora.py，完全从0实现LoRA流程，不依赖第三方库的封装。
function trainlora() {
    # lora="lora_medical"
    lora="lora_identity"
    python train_lora.py \
        --epochs 1 \
        --batch_size 8 \
        --device mps \
        --dtype float32 \
        --dim $dim \
        --log_interval 50 \
        --save_interval 1 \
        --data_path ./dataset/"$lora".jsonl \
        --base_model full_sft \
        --lora_name "$lora"
}

function eval_lora() {
    # lora="lora_medical"
    lora="lora_identity"

    python eval_model.py \
        --lora_name "$lora" \
        --device mps \
        --dim $dim \
        --model_mode 1
}

dim=512

cmd="$1"

[ -z "$cmd" ] && echo "bash train.bash <cmd>" && exit 1

$cmd


