import torch
import argparse
import onmt
import onmt.model_builder
import onmt.inputters
import onmt.opts

from onmt.utils.misc import use_gpu
from onmt.utils.logging import init_logger, logger

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-fields', required=True,
                    help='Path to fields .pt file')
parser.add_argument('-output_dir', default='.',
                    help="""Path to output the embeddings""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def write_embeddings(filename, dict, embeddings):
    with open(filename, 'wb') as file:
        for i in range(min(len(embeddings), len(dict.itos))):
            str = dict.itos[i].encode("utf-8")
            for j in range(len(embeddings[0])):
                str = str + (" %5f" % (embeddings[i][j])).encode("utf-8")
            file.write(str + b"\n")


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Add in default model arguments, possibly added since training.
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    fields = torch.load(opt.fields)
    model_opt = checkpoint['opt']

    src_dicts = [f.vocab for n, f in fields['src']]
    tgt_dicts = [f.vocab for n, f in fields['tgt']]
    src_dict, tgt_dict = src_dicts[0], tgt_dicts[0]
    src_feat_dicts, tgt_feat_dicts = src_dicts[1:], tgt_dicts[1:]

    model_opt = checkpoint['opt']
    for arg in dummy_opt.__dict__:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt.__dict__[arg]

    model = onmt.model_builder.build_base_model(
        model_opt, src_dict, src_feat_dicts, tgt_dict, tgt_feat_dicts,
        use_gpu(opt), checkpoint)
    encoder = model.encoder
    decoder = model.decoder

    encoder_embeddings = encoder.embeddings.word_lut.weight.data.tolist()
    decoder_embeddings = decoder.embeddings.word_lut.weight.data.tolist()

    logger.info("Writing source embeddings")
    write_embeddings(opt.output_dir + "/src_embeddings.txt", src_dict,
                     encoder_embeddings)

    logger.info("Writing target embeddings")
    write_embeddings(opt.output_dir + "/tgt_embeddings.txt", tgt_dict,
                     decoder_embeddings)

    logger.info('... done.')
    logger.info('Converting model...')


if __name__ == "__main__":
    init_logger('extract_embeddings.log')
    main()
