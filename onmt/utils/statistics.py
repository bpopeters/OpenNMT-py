""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys

from torch.distributed import get_rank
from onmt.utils.distributed import all_gather_list
from onmt.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0, **kwargs):
        self.stats = kwargs
        self.stats['loss'] = loss
        self.stats['n_words'] = n_words
        self.stats['n_correct'] = n_correct
        self.stats['n_src_words'] = 0
        self.start_time = time.time()

    @property
    def loss(self):
        return self.stats['loss']

    @property
    def n_words(self):
        return self.stats['n_words']

    @property
    def n_correct(self):
        return self.stats['n_correct']

    @property
    def n_src_words(self):
        return self.stats['n_src_words']

    def add_src_lengths(self, src_lengths):
        self.stats['n_src_words'] += src_lengths

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    # the job here is essentially to merge two dictionaries with
    # identical keys
    def update(self, other, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            other: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        #assert self.stats.keys() == other.stats.keys()  # actually correct?
        for k, v in other.stats.items():
            if k in self.stats:
                if k != 'n_src_words' or update_n_src_words:
                    self.stats[k] += v
            else:
                self.stats[k] = v

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        acc = self.accuracy()
        # wouldn't it be better if I could assemble this dynamically?
        """
        metrics = ["Step", "acc", "ppl", "xent", "lr"]
        "; ".join([])
        report_format = "Step {step}/{num_steps}; acc: {acc}; ppl: {ppl}; " +
                        "xent: %4.2f; lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec"
        """
        metrics = "Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; lr: %7.5f; " % \
            (step, num_steps, self.accuracy(), self.ppl(), self.xent(), learning_rate)
        if 'n_supported' in self.stats:
            metrics += "supp: {:.2f} ; ".format(self.stats['n_supported'] / self.n_words)
        time_metrics = "%3.0f/%3.0f tok/s; %6.0f sec" % \
            (self.n_src_words / (t + 1e-5), self.n_words / (t + 1e-5),
             time.time() - start)
        logger.info(metrics + time_metrics)
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
