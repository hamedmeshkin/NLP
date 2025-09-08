import os
import csv
from torch.utils.data import DataLoader, Dataset

class InputExample(object):  #from BERT code
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class myDataSet(Dataset):
  def __init__(self, X):  #here X is a list of objects of class InputExample
    self.X = X

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    sentence = self.X[index].text_a
    #label = int(self.X[index].label)   #sometimes labels are int sometimes strings
    label = self.X[index].label
    return sentence, label


class DataProcessor(object):  #from bert code
    """Base class for processing data sets and applying various prompts."""

    def get_train_examples(self, data_file):
          """base class."""
          return self._create_examples(  #these are leading examples for in-context learning
                                         self._read_tsv(data_file), "train")

    def get_dev_examples(self, data_file):
        """base class."""
        return self._create_examples(
            self._read_tsv(data_file), "dev")

    def get_test_examples(self, data_file):
        """base class."""
        return self._create_examples(
            self._read_tsv(data_file), "test")


    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_train_examples(self, data_file):
        """base class."""
        return self._create_examples(  #these are leading examples for in-context learning
                                       self._read_tsv(data_file), "train")

    def get_dev_examples(self, data_file):
        """base class."""
        return self._create_examples(
            self._read_tsv(data_file), "dev")

    def get_imblearner_examples(self, lines):
        """base class."""
        return self._create_examples2(lines, "dev")

    def get_test_examples(self, data_file):
        """base class."""
        return self._create_examples(
            self._read_tsv(data_file), "test")

    def _create_examples(self, lines, set_type):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None): #new csv module, empty quotechar not allowed?
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class Superglue_RTE_shouldassume_Processor(DataProcessor):
    """from https://huggingface.co/spaces/bigscience/promptsource
      dataset: super_glue
      subset: RTE (Recognizing Textual Entailment)
      promp name: should assume
      reference: Webson & Pavlick 2021"""
    """from the T0 paper: page 49
      note that per Table 5 of T0 paper this dataset/subset was used for evaluation, not training"""

    def __init__(self, assumesentence = 'should we assume "this sentence is about pharmacokinetic drug-drug interaction" is true? Yes or no?'):
        self.assumesentence = assumesentence

    def get_labels(self):
          """See base class."""
          return ["0", "1"]

    def _create_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #right now these files inherit a weird thing from before that "dev" have different column order than others
        guid = "%s-%s" % (set_type, i)
        if set_type == "dev":
          text_a = 'Given "' + line[1] + '", ' + self.assumesentence + '\n'   #prefix the leading examples
          label = line[0]
        else:
          text_a = 'Given "' + line[0] + '", ' + self.assumesentence +'\n'
          label = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples



    def create_leading_examples(self, input_examples, separator = "###"):
        leading_examples = ""

        for (i, example) in enumerate(input_examples):
            answer = "Yes."
            if example.label == "0":
                answer = "No."
            leading_examples = leading_examples + example.text_a + answer + "\n" + separator + "\n"
        return leading_examples


class AI2ARC_Challenge_qaoptions_Processor(DataProcessor):
    """from https://huggingface.co/spaces/bigscience/promptsource
    dataset: ai2_arc
    subset: ARC_Challenge
    promp name: qa_options
    reference: """
    """from the T0 paper: page 66
    note that per Table 5 of T0 paper, this dataset/subset was used for training for T0pp
    """

    def get_labels(self):
          """See base class."""
          return ["Non-Pharmacokinetic", "Pharmacokinetic"]

    def _create_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #the intended files are /home/lizhi/documents/AI_CORD19/DDI_data/corpus/TAC_DDI2019/TACDDI_nomask_train or dev
        #they both have the first line as column names: source sentence DDItype
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        if set_type == "dev" or set_type == "train":
          text_a = (line[1] + '. Which drug-drug interaction type is this sentence about?\n'+
                    'Options:\n'+
                    '- Pharmacokinetic\n'+
                    '- Non-Pharmacokinetic')

          label = line[2]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_examples2(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #the intended files are /home/lizhi/documents/AI_CORD19/DDI_data/corpus/TAC_DDI2019/TACDDI_nomask_train or dev
        #they both have the first line as column names: source sentence DDItype
        guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def create_leading_examples(self, input_examples, separator = "###"):
        leading_examples = ""

        for (i, example) in enumerate(input_examples):
            answer = example.label
            if example.label == "Pharmacokinetic interaction":
                answer = "Pharmacokinetic"
            elif example.label == "Non-Pharmacokinetic interaction":
                answer = "Non-Pharmacokinetic"

            leading_examples = leading_examples + example.text_a + "\n" + answer + "\n" + separator + "\n"
        return leading_examples


class QASC_qawithseparatedfacts1_Processor(DataProcessor):
    """from https://huggingface.co/spaces/bigscience/promptsource
    dataset: QASC (question answering with separated facts)
    subset:
    promp name: qa_with_separated_facts_1
    reference: """
    """from the T0 paper: page 121
    note that per Table 5 of T0 paper, this dataset/subset was used for training for T0pp
    """
    def get_facts(self):
        facts = ("Some sentences are about pharmacokinetic drug-drug interaction, where a drug alters the disposition " +
                  "(absorption, distribution, elimination) of another drug, usually resulting in the change of plasma drug concentrations. " +
                  "Some sentences are about pharmacodynamic drug-drug interaction, where a drug alters the pharmacological effects of another drug, " +
                  "usually without changing the plasma drug concentrations. " +
                  "Some sentences are about nonspecific interaction, where a drug has an effect on another drug, but it is unclear if it is pharmcokinetic or pharmacodynamic drug-drug interaction. "+
                  "Some sentences are not about drug-drug interaction at all. ")
        return facts

    def get_labels(self):
          """See base class."""
          return ["Non-Pharmacokinetic", "Pharmacokinetic"]

    def _create_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #the intended files are /home/lizhi/documents/AI_CORD19/DDI_data/corpus/TAC_DDI2019/TACDDI_nomask_train or dev
        #or multiclass_each20_dev
        #they both have the first line as column names: source sentence DDItype
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        if set_type == "dev" or set_type == "train":
          text_a = (self.get_facts() + 'Given these facts, what is this the sentence "' + line[1] + '" about among the following options:\n'+
                    '- Pharmacokinetic\n'+
                    '- Non-Pharmacokinetic')

          label = line[2]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def create_leading_examples(self, input_examples):
        leading_examples = self.get_facts()
        return leading_examples


class SCIQ_multiplechoice_Processor(DataProcessor):
    """from https://huggingface.co/spaces/bigscience/promptsource
    dataset: SCIQ
    subset:
    promp name: Multiple Choice
    reference: """
    """from the T0 paper: page 143
    note that per Table 5 of T0 paper, this dataset/subset was used for training for T0pp
    """
    def get_facts(self):
        facts = ("Some sentences are about pharmacokinetic drug-drug interaction, where a drug alters the disposition " +
                  "(absorption, distribution, elimination) of another drug, usually resulting in the change of plasma drug concentrations. " +
                  "Some sentences are about pharmacodynamic drug-drug interaction, where a drug alters the pharmacological effects of another drug, " +
                  "usually without changing the plasma drug concentrations. " +
                  "Some sentences are about unspecified interaction, where a drug has an effect on another drug, but it is unclear if it is pharmcokinetic or pharmacodynamic drug-drug interaction. "+
                  "Some sentences are not about drug-drug interaction at all. ")
        return facts

    def get_labels(self):
          """See base class."""
          return ["No interaction", "Unspecified interaction", "Pharmacodynamic", "Pharmacokinetic"]

    def _create_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #the intended files are /home/lizhi/documents/AI_CORD19/DDI_data/corpus/TAC_DDI2019/TACDDI_nomask_train or dev
        #or multiclass_each20_dev
        #they both have the first line as column names: source sentence DDItype
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        if set_type == "dev" or set_type == "train":
          text_a = ('Answer the following question given this paragraph:\n' +
                    self.get_facts() + '\n' +
                    'Q: which drug-drug interaction type is the sentence "' + line[1] + '" about?\n' +
                    'Choices:\n' +
                    '- Pharmacokinetic\n'+
                    '- Non-Pharmacokinetic\n'+
                    'A: ')

          label = line[2]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def create_leading_examples(self, input_examples):
        leading_examples = self.get_facts()
        return leading_examples



class TK_instruct_def_pos_Processor(DataProcessor):


    def __init__(self, definition = 'In this task, you are given two sentences. Indicate if the first sentence clearly entails the second sentence (i.e., one can conclude the 2nd sentence by reading the 1st one). Indicate your answer with â€œ1â€� if the first sentence entails the second sentence, otherwise answer with â€œ0â€�.'):
        self.definition = definition

    def get_labels(self):
          """See base class."""
          return ["0", "1"]

    def _create_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #right now these files inherit a weird thing from before that "dev" have different column order than others
        guid = "%s-%s" % (set_type, i)
        if set_type == "dev":
          text_a = 'Input: Sentence 1: ' + line[1]  + ' Sentence 2: This is about pharmacokinetic drug-drug interaction. \nOutput: '
          label = line[0]
        else:
          text_a = 'Input: Sentence 1: ' + line[0]  + ' Sentence 2: This is about pharmacokinetic drug-drug interaction. \nOutput: '
          label = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def create_leading_examples(self, input_examples, separator = "\n"):
        leading_examples = self.definition + '\n'

        for (i, example) in enumerate(input_examples):
            leading_examples = leading_examples + example.text_a + example.label + separator
        return leading_examples


class Flan_T5_QA_Processor(DataProcessor):

    def __init__(self, assumesentence = 'should we assume "this sentence is about pharmacokinetic drug-drug interaction" is true?'):
        self.assumesentence = assumesentence

    def get_labels(self):
          """See base class."""
          return ["0", "1"]

    def _create_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #right now these files inherit a weird thing from before that "dev" have different column order than others
        guid = "%s-%s" % (set_type, i)
        if set_type == "dev":
          text_a = 'Q: Answer the following yes/no question. \nGiven "' + line[1] + '", ' + self.assumesentence + '\nA: '   #prefix the leading examples
          label = line[0]
        else:
          text_a = 'Q: Answer the following yes/no question. \nGiven "' + line[0] + '", ' + self.assumesentence + '\nA: '
          label = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples



    def create_leading_examples(self, input_examples, separator = "\n"):
        leading_examples = ""

        for (i, example) in enumerate(input_examples):
            answer = "Yes."
            if example.label == "0":
                answer = "No."
            leading_examples = leading_examples + example.text_a + answer + separator
        return leading_examples


class AI2ARC_Intrinsic_Challenge_qaoptions_Processor(DataProcessor):
    """from https://huggingface.co/spaces/bigscience/promptsource
    dataset: ai2_arc
    subset: ARC_Challenge
    promp name: qa_options
    reference: """
    """from the T0 paper: page 66
    note that per Table 5 of T0 paper, this dataset/subset was used for training for T0pp
    """

    def get_labels(self):
          """See base class."""
          return ["Non-Intrinsic", "Intrinsic"]

    def _create_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #the intended files are /home/lizhi/documents/AI_CORD19/DDI_data/corpus/TAC_DDI2019/TACDDI_nomask_train or dev
        #they both have the first line as column names: source sentence DDItype
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        if set_type == "dev" or set_type == "train":
            question = "Is it true that the given sentence relates exclusively to the drug's pharmacokinetics as altered by patients' intrinsic factors, such as Gender, Age, Weight, Genetics, Organ's function, Additional diseases, and so forth?"
            text_a = (line[1] + '\n' + question + " \n" +
                      'Just answer with Yes or No')

            label = line[0]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_examples2(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #the intended files are /home/lizhi/documents/AI_CORD19/DDI_data/corpus/TAC_DDI2019/TACDDI_nomask_train or dev
        #they both have the first line as column names: source sentence DDItype
        guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def create_leading_examples(self, input_examples, separator = "###"):
        leading_examples = ""

        for (i, example) in enumerate(input_examples):
            answer = example.label
            if example.label == "Intrinsic factor":
                answer = "Yes"
            elif example.label == "Non-intrinsic factor":
                answer = "No"

            leading_examples = leading_examples + example.text_a + "\n" + answer + "\n" + separator + "\n"
        return leading_examples
