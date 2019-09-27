""" Copyright (c) 2019 NVIDIA Corporation
See the tutorial and download the data here:
https://nvidia.github.io/NeMo/nlp/
neural-machine-translation.html#translation-with-pretrained-model
"""
import torch

import nemo
from nemo.utils.lr_policies import get_lr_policy

import nemo_nlp
from nemo_nlp.utils.callbacks.translation import \
    eval_iter_callback, eval_epochs_done_callback

parser = nemo.utils.NemoArgParser(description='Transformer for WMT16 En-De')
parser.set_defaults(train_dataset="train",
                    eval_datasets=["valid"],
                    work_dir="outputs/transformer_nmt",
                    optimizer="novograd",
                    num_epochs=500,  # 5 is too small - only use for test
                    batch_size=4096,
                    eval_batch_size=4096,
                    lr_policy='CosineAnnealing',
                    lr=0.005,
                    beta1=0.95,
                    beta2=0.25,
                    weight_decay=0,
                    max_steps=50000,
                    iter_per_step=1,
                    eval_freq=1000)
parser.add_argument("--data_dir", default="../../tests/data/en_de", type=str)
parser.add_argument("--dataset_name", default="wmt16", type=str)
parser.add_argument("--src_lang", default="en", type=str)
parser.add_argument("--tgt_lang", default="de", type=str)
parser.add_argument("--d_model", default=512, type=int)
parser.add_argument("--d_inner", default=2048, type=int)
parser.add_argument("--num_layers", default=4, type=int)
parser.add_argument("--num_attn_heads", default=8, type=int)
parser.add_argument("--embedding_dropout", default=0.2, type=float)
parser.add_argument("--ffn_dropout", default=0.2, type=float)
parser.add_argument("--attn_score_dropout", default=0.2, type=float)
parser.add_argument("--attn_layer_dropout", default=0.2, type=float)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--label_smoothing", default=0.1, type=float)
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--tokenizer_model", default="bpe8k_yttm.model", type=str)
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)

args = parser.parse_args()

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=args.work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__])
"""Define tokenizer
In this example, we use YouTokenToMe tokenizer trained on joint
English & German data for both source and target languages.

When the src and tgt languages are very different, it's better to use separate
tokenizers.
"""
tokenizer = nemo_nlp.YouTokenToMeTokenizer(
    model_path=f"{args.data_dir}/{args.tokenizer_model}")
vocab_size = tokenizer.vocab_size

# instantiate necessary modules for the whole translation pipeline, namely
# data layers, encoder, decoder, output log_softmax, beam_search_translator
# and loss function
encoder = nemo_nlp.TransformerEncoderNM(
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    embedding_dropout=args.embedding_dropout,
    num_attn_heads=args.num_attn_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=vocab_size,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_seq_length)

decoder = nemo_nlp.TransformerDecoderNM(
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    embedding_dropout=args.embedding_dropout,
    num_attn_heads=args.num_attn_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=vocab_size,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_seq_length)

log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
    vocab_size=vocab_size,
    d_model=args.d_model)

beam_search = nemo_nlp.BeamSearchTranslatorNM(
    decoder=decoder,
    log_softmax=log_softmax,
    max_seq_length=args.max_seq_length,
    beam_size=args.beam_size,
    bos_token=tokenizer.bos_id(),
    pad_token=tokenizer.pad_id(),
    eos_token=tokenizer.eos_id())

loss_fn = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(
    pad_id=tokenizer.pad_id(),
    label_smoothing=args.label_smoothing)

# tie weight of embedding and log_softmax layers
log_softmax.log_softmax.dense.weight = \
    encoder.embedding_layer.token_embedding.weight
decoder.embedding_layer.token_embedding.weight = \
    encoder.embedding_layer.token_embedding.weight


def create_pipeline(dataset_src,
                    dataset_tgt,
                    tokens_in_batch,
                    clean=True,
                    training=True):
    dataset = nemo_nlp.TranslationDataset(tokenizer_src=tokenizer,
                                          tokenizer_tgt=tokenizer,
                                          dataset_src=dataset_src,
                                          dataset_tgt=dataset_tgt,
                                          tokens_in_batch=tokens_in_batch,
                                          clean=clean)
    data_layer = nemo_nlp.TranslationDataLayer(dataset)
    src, src_mask, tgt, tgt_mask, labels, sent_ids = data_layer()
    src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
    tgt_hiddens = decoder(input_ids_tgt=tgt,
                          hidden_states_src=src_hiddens,
                          input_mask_src=src_mask,
                          input_mask_tgt=tgt_mask)
    logits = log_softmax(hidden_states=tgt_hiddens)
    loss = loss_fn(log_probs=logits, target_ids=labels)
    beam_results = None
    if not training:
        beam_results = beam_search(hidden_states_src=src_hiddens,
                                   input_mask_src=src_mask)
    return loss, [tgt, loss, beam_results, sent_ids]


train_dataset_src = f"{args.data_dir}/{args.train_dataset}.{args.src_lang}"
train_dataset_tgt = f"{args.data_dir}/{args.train_dataset}.{args.tgt_lang}"

# train_dataset_src = f"{args.data_dir}/{args.eval_datasets[0]}.{args.src_lang}"
# train_dataset_tgt = f"{args.data_dir}/{args.eval_datasets[0]}.{args.tgt_lang}"

train_loss, _ = create_pipeline(train_dataset_src,
                                train_dataset_tgt,
                                args.batch_size,
                                clean=True)

eval_dataset_src = f"{args.data_dir}/{args.eval_datasets[0]}.{args.src_lang}"
eval_dataset_tgt = f"{args.data_dir}/{args.eval_datasets[0]}.{args.tgt_lang}"

eval_loss, eval_tensors = create_pipeline(eval_dataset_src,
                                          eval_dataset_tgt,
                                          args.eval_batch_size,
                                          clean=True,
                                          training=False)

# callback which prints training loss once in a while
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=100,
    print_func=lambda x: str(x[0].item()),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer)

# callback which calculates evaluation loss and both common BLEU and SacreBLEU
# scores between outputs of beam search and reference translations
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(x, y, tokenizer),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, validation_dataset=eval_dataset_tgt),
    eval_step=args.eval_freq,
    tb_writer=nf.tb_writer)

# callback which saves checkpoints once in a while
ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    epoch_freq=args.save_epoch_freq,
    step_freq=args.save_step_freq,
    checkpoints_to_keep=1)

# define and launch training algorithm (optimizer)
max_num_epochs = 0 if args.interactive else args.num_epochs

callbacks = [ckpt_callback]

if not args.interactive:
    callbacks.extend([train_callback, eval_callback])

# define learning rate decay policy
lr_policy_fn = get_lr_policy(args.lr_policy,
                             total_steps=args.max_steps,
                             warmup_steps=args.warmup_steps)

nf.train(tensors_to_optimize=[train_loss],
         callbacks=callbacks,
         optimizer=args.optimizer,
         lr_policy=lr_policy_fn,
         optimization_params={"num_epochs": max_num_epochs,
                              "lr": args.lr,
                              "weight_decay": args.weight_decay,
                              "betas": (args.beta1, args.beta2)},
         batches_per_step=args.iter_per_step)


def translate_sentence(text):
    """ helper function that takes source text as input, tokenizes it, and
    output its translation
    """
    ids = tokenizer.text_to_ids(text)
    ids = [tokenizer.bos_id()] + ids + [tokenizer.eos_id()]
    ids_tensor = torch.Tensor(ids).long().to(encoder._device).unsqueeze(0)
    ids_mask = torch.ones_like(ids_tensor).half()
    encoder_states = encoder.forward(ids_tensor, ids_mask).half()
    translation_ids = beam_search.forward(encoder_states, ids_mask)
    ids_list = list(translation_ids.detach().cpu().numpy()[0])
    translation_text = tokenizer.ids_to_text(ids_list)
    return translation_text


# code below was not covered in the tutorial, it is used to generate
# translations on the fly with pre-trained model
if args.interactive:
    # set all modules into evaluation mode (turn off dropout)
    encoder.eval()
    decoder.eval()
    log_softmax.eval()

    print("========== Interactive translation mode ==========")
    input_text = 'anything'
    while input_text.strip():
        input_text = input(f'Text in {args.src_lang} to be translated: ')
        print('Translated:', translate_sentence(input_text.strip()))
