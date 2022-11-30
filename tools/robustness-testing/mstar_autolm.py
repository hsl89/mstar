import gc
from typing import Optional
from mstar import AutoModel 
from mstar import AutoTokenizer
from mstar import AutoConfig
from lm_eval.models.huggingface import AutoCausalLM, AutoSeq2SeqLM
from types import MethodType
from tokenizers.processors import TemplateProcessing
from torch.cuda import empty_cache

class MStarAutoCausalLM(AutoCausalLM):
    """Causal language modeling. Extends the huggingface.AutoCausalLM class to support mstar models.
    """
    AUTO_MODEL_CLASS = AutoModel
    AUTO_TOKENIZER_CLASS = AutoTokenizer
    AUTO_CONFIG_CLASS = AutoConfig
    
    @property
    def add_special_tokens(self) -> bool:
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False

class MStarAutoSeq2SeqLM(AutoSeq2SeqLM):
    """Seq2Seq language modeling. Extends the huggingface.AutoSeq2SeqLM class to support mstar models.
    """
    AUTO_MODEL_CLASS = AutoModel
    AUTO_TOKENIZER_CLASS = AutoTokenizer
    AUTO_CONFIG_CLASS = AutoConfig

    def __init__(
        self,
        pretrained: str,
        special_prefix_token: Optional[str] = None,
        special_prefix_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(pretrained, **kwargs)

        if special_prefix_token_id != None:
            assert isinstance(special_prefix_token_id, int), "special_prefix_token_id is not an int"
        
        if special_prefix_token != None and special_prefix_token_id != None:
                assert [special_prefix_token_id] == self.tokenizer.convert_tokens_to_ids([special_prefix_token]), "special_prefix_token does not map to special_prefix_token_id"
        
        if special_prefix_token:
            special_prefix_token_id = self.tokenizer.convert_tokens_to_ids([special_prefix_token])[0]

        if special_prefix_token_id != None:
            if type(self.tokenizer).__name__[-4:] == 'Fast':
                print("Rust tokenizer detected. Overriding...")
                self.override_build_inputs_with_special_tokens_rust(special_prefix_token_id)
            else:
                print("Python tokenizer detected. Overriding...")
                self.override_build_inputs_with_special_tokens_python(special_prefix_token_id)

    def override_build_inputs_with_special_tokens_rust(self, special_prefix_token_id):
        special_prefix_token = self.tokenizer._tokenizer.id_to_token(special_prefix_token_id)
        
        new_pp = TemplateProcessing(
            single=f"{special_prefix_token} $A:0 {self.tokenizer.eos_token}", 
            pair=f"{special_prefix_token} $A:0 $B:0 {self.tokenizer.eos_token}", 
            special_tokens=[(special_prefix_token, special_prefix_token_id), (self.tokenizer.eos_token, self.tokenizer.eos_token_id)]
        )

        self.tokenizer._tokenizer.post_processor = new_pp
        print("Override complete:", self.tokenizer.encode("Override complete", add_special_tokens=True))

    def _create_auto_model(
        self,
        *,
        pretrained,
        revision,
        subfolder,
        device_map=None,
        max_memory=None,
        offload_folder=None,
        torch_dtype=None,
    ):
        """Overrides superclass function to pass `softmax_type='torch'` argument.
        """
        model = AutoModel.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            torch_dtype=torch_dtype,
            softmax_type='torch',
        )
        return model

    def _model_call(self, inputs, labels=None):
        gc.collect()
        empty_cache()
        return super()._model_call(inputs, labels)

    def _model_generate(self, inputs, max_tokens, stop):
        gc.collect()
        empty_cache()
        return super()._model_generate(inputs, max_tokens, stop)

    def override_build_inputs_with_special_tokens_python(self, special_prefix_token_id):
        self.tokenizer.special_prefix_token_id = special_prefix_token_id
      
        def custom_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            token_ids = token_ids_0.copy()
            if token_ids_1 is not None:
                token_ids += token_ids_1
            return [self.special_prefix_token_id] + token_ids + [self.eos_token_id]
            
        self.tokenizer.build_inputs_with_special_tokens = MethodType(custom_build_inputs_with_special_tokens, self.tokenizer)

        print("Override complete:", self.tokenizer.encode("Override complete", add_special_tokens=True))#['input_ids'])

    @property
    def add_special_tokens(self) -> bool:
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return True


class AutoUL2(AutoSeq2SeqLM):
    """Custom class for inserting UL2 mode tokens as a prefix to the input and <extra_id_0> as a suffix to the input
    """

    def __init__(
        self,
        pretrained: str,
        mode_prefix_token: Optional[str] = None,
        add_extra_id_0: Optional[int] = False,
        **kwargs,
    ):
        super().__init__(pretrained, **kwargs)
        print('mode_prefix_token', mode_prefix_token)
        if mode_prefix_token:
            mode_token_ids = self.tokenizer.encode(mode_prefix_token, add_special_tokens=False)
            print("Python tokenizer detected. Overriding...")
            self.override_build_inputs_with_special_tokens_python(mode_token_ids, add_extra_id_0=add_extra_id_0)

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ):
        """Copied from huggingface.py to set use_fast to False 
        """
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _model_call(self, inputs, labels=None):
        gc.collect()
        empty_cache()
        return super()._model_call(inputs, labels)

    def _model_generate(self, inputs, max_tokens, stop):
        gc.collect()
        empty_cache()
        return super()._model_generate(inputs, max_tokens, stop)

    def override_build_inputs_with_special_tokens_python(self, mode_token_ids, add_extra_id_0=False):
        self.tokenizer.mode_token_ids = mode_token_ids
        if add_extra_id_0:
            self.tokenizer.extra_id_0 = [self.tokenizer.convert_tokens_to_ids('<extra_id_0>')]
        else:
            self.tokenizer.extra_id_0 = []
      
        def custom_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            token_ids = token_ids_0.copy()
            if token_ids_1 is not None:
                token_ids += token_ids_1
            return self.mode_token_ids + token_ids + self.extra_id_0 + [self.eos_token_id]
            
        self.tokenizer.build_inputs_with_special_tokens = MethodType(custom_build_inputs_with_special_tokens, self.tokenizer)

        print("Override complete:", self.tokenizer.encode("Override complete", add_special_tokens=True))#['input_ids'])

    @property
    def add_special_tokens(self) -> bool:
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return True