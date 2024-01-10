import tiktoken
import fasttext
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import Counter
import re
import os


class Labler:

    def __init__(self):
        device = int(os.getenv("CUDA_DEVICE", -1))
        fasttext_model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.encoder = tiktoken.encoding_for_model("gpt2")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", device=device, truncation=True)
        self.iso_639_2_mapping = {'ace_Arab': 'ace', 'ace_Latn': 'ace', 'acm_Arab': 'acm', 'acq_Arab': 'aeb', 'aeb_Arab': 'aeb', 'afr_Latn': 'afr', 'ajp_Arab': 'ajp', 'aka_Latn': 'aka', 'amh_Ethi': 'amh', 'apc_Arab': 'apc', 'arb_Arab': 'ara', 'arb_Latn': 'ara', 'ars_Arab': 'ars', 'ary_Arab': 'ary', 'arz_Arab': 'arz', 'asm_Beng': 'asm', 'ast_Latn': 'ast', 'awa_Deva': 'awa', 'ayr_Latn': 'ayr', 'azb_Arab': 'azb', 'azj_Latn': 'azj', 'bak_Cyrl': 'bak', 'bam_Latn': 'bam', 'ban_Latn': 'ban', 'bel_Cyrl': 'bel', 'bem_Latn': 'bem', 'ben_Beng': 'ben', 'bho_Deva': 'bho', 'bjn_Arab': 'bjn', 'bjn_Latn': 'bjn', 'bod_Tibt': 'bod', 'bos_Latn': 'bos', 'bug_Latn': 'bug', 'bul_Cyrl': 'bul', 'cat_Latn': 'cat', 'ceb_Latn': 'ceb', 'ces_Latn': 'ces', 'cjk_Latn': 'cjk', 'ckb_Arab': 'ckb', 'crh_Latn': 'crh', 'cym_Latn': 'cym', 'dan_Latn': 'dan', 'deu_Latn': 'deu', 'dik_Latn': 'din', 'dyu_Latn': 'dyu', 'dzo_Tibt': 'dzo', 'ell_Grek': 'ell', 'eng_Latn': 'eng', 'epo_Latn': 'epo', 'est_Latn': 'est', 'eus_Latn': 'eus', 'ewe_Latn': 'ewe', 'fao_Latn': 'fao', 'fij_Latn': 'fij', 'fin_Latn': 'fin', 'fon_Latn': 'fon', 'fra_Latn': 'fra', 'fur_Latn': 'fur', 'fuv_Latn': 'ful', 'gla_Latn': 'gla', 'gle_Latn': 'gle', 'glg_Latn': 'glg', 'grn_Latn': 'grn', 'guj_Gujr': 'guj', 'hat_Latn': 'hat', 'hau_Latn': 'hau', 'heb_Hebr': 'heb', 'hin_Deva': 'hin', 'hne_Deva': 'hne', 'hrv_Latn': 'hrv', 'hun_Latn': 'hun', 'hye_Armn': 'hye', 'ibo_Latn': 'ibo', 'ilo_Latn': 'ilo', 'ind_Latn': 'ind', 'isl_Latn': 'isl', 'ita_Latn': 'ita', 'jav_Latn': 'jav', 'jpn_Jpan': 'jpn', 'kab_Latn': 'kab', 'kac_Latn': 'kac', 'kam_Latn': 'kam', 'kan_Knda': 'kan', 'kas_Arab': 'kas', 'kas_Deva': 'kas', 'kat_Geor': 'kat', 'knc_Arab': 'kau', 'knc_Latn': 'kau', 'kaz_Cyrl': 'kaz', 'kbp_Latn': 'kbp', 'kea_Latn': 'kea', 'khm_Khmr': 'khm', 'kik_Latn': 'kik', 'kin_Latn': 'kin', 'kir_Cyrl': 'kir', 'kmb_Latn': 'kmb', 'kmr_Latn': 'kmr', 'kon_Latn': 'kon', 'kor_Hang': 'kor', 'lao_Laoo': 'lao', 'lij_Latn': 'lij',
                                  'lim_Latn': 'lim', 'lin_Latn': 'lin', 'lit_Latn': 'lit', 'lmo_Latn': 'lmo', 'ltg_Latn': 'ltg', 'ltz_Latn': 'ltz', 'lua_Latn': 'lub', 'lug_Latn': 'lug', 'luo_Latn': 'luo', 'lus_Latn': 'lus', 'lvs_Latn': 'lav', 'mag_Deva': 'mag', 'mai_Deva': 'mai', 'mal_Mlym': 'mal', 'mar_Deva': 'mar', 'min_Arab': 'min', 'min_Latn': 'min', 'mkd_Cyrl': 'mkd', 'plt_Latn': 'plt', 'mlt_Latn': 'mlt', 'mni_Beng': 'mni', 'khk_Cyrl': 'mon', 'mos_Latn': 'mos', 'mri_Latn': 'mri', 'mya_Mymr': 'mya', 'nld_Latn': 'nld', 'nno_Latn': 'nno', 'nob_Latn': 'nob', 'npi_Deva': 'npi', 'nso_Latn': 'nso', 'nus_Latn': 'nus', 'nya_Latn': 'nya', 'oci_Latn': 'oci', 'gaz_Latn': 'orm', 'ory_Orya': 'ori', 'pag_Latn': 'pag', 'pan_Guru': 'pan', 'pap_Latn': 'pap', 'pes_Arab': 'pes', 'pol_Latn': 'pol', 'por_Latn': 'por', 'prs_Arab': 'prs', 'pbt_Arab': 'pst', 'quy_Latn': 'quy', 'ron_Latn': 'ron', 'run_Latn': 'run', 'rus_Cyrl': 'rus', 'sag_Latn': 'sag', 'san_Deva': 'san', 'sat_Olck': 'sat', 'scn_Latn': 'scn', 'shn_Mymr': 'shn', 'sin_Sinh': 'sin', 'slk_Latn': 'slk', 'slv_Latn': 'slv', 'smo_Latn': 'smo', 'sna_Latn': 'sna', 'snd_Arab': 'snd', 'som_Latn': 'som', 'sot_Latn': 'sot', 'spa_Latn': 'spa', 'als_Latn': 'sqi', 'srd_Latn': 'srd', 'srp_Cyrl': 'srp', 'ssw_Latn': 'ssw', 'sun_Latn': 'sun', 'swe_Latn': 'swe', 'swh_Latn': 'swa', 'szl_Latn': 'szl', 'tam_Taml': 'tam', 'tat_Cyrl': 'tat', 'tel_Telu': 'tel', 'tgk_Cyrl': 'tgk', 'tgl_Latn': 'tgl', 'tha_Thai': 'tha', 'tir_Ethi': 'tir', 'taq_Latn': 'tmh', 'taq_Tfng': 'tmh', 'tpi_Latn': 'tpi', 'tsn_Latn': 'tsn', 'tso_Latn': 'tso', 'tuk_Latn': 'tuk', 'tum_Latn': 'tum', 'tur_Latn': 'tur', 'twi_Latn': 'twi', 'tzm_Tfng': 'tzm', 'uig_Arab': 'uig', 'ukr_Cyrl': 'ukr', 'umb_Latn': 'umb', 'urd_Arab': 'urd', 'uzn_Latn': 'uzn', 'vec_Latn': 'vec', 'vie_Latn': 'vie', 'war_Latn': 'war', 'wol_Latn': 'wol', 'xho_Latn': 'xho', 'ydd_Hebr': 'ydd', 'yor_Latn': 'yor', 'yue_Hant': 'yue', 'zho_Hans': 'zho', 'zho_Hant': 'zho', 'zsm_Latn': 'zsm', 'zul_Latn': 'zul'}
        self.fasttext_model = fasttext.load_model(fasttext_model_path)

        ner_tokenizer = AutoTokenizer.from_pretrained(
            "dslim/bert-base-NER", model_max_len=512)
        ner_model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER")

        self.ner_pipeline = pipeline(
            "ner", model=ner_model, tokenizer=ner_tokenizer, device=device)
        self.ner_mapping = {
            "O": None,
            "B-MISC": None,
            "I-MISC": None,
            "B-PER": "human",
            "I-PER": "human",
            "B-ORG": "company",
            "I-ORG": "company",
            "B-LOC": "location",
            "I-LOC": "location",
        }

    def get_tags(self, event):

        if not self.is_root(event["tags"]):
            return []

        tags = []
        tags.append(["L", "nostr.event.kind:root-note"])

        tags.append(["L", f"nostr.event.content:length-range:{self.__content_length_to_range(len(event['content']))}"])
        tags.append(["L", f"nostr.event.content:length-range-tokens:gpt2:{self.__content_length_to_range(len(self.encoder.encode(event['content'])))}"])

        language_label, language_score = self.__get_lang(event["content"])
        if language_score > 0.5 and language_label:
            tags.append(["L", f"nostr.event.content:iso-639-2:{language_label}"])

        sentiment = self.sentiment_pipeline(event["content"])[0]
        if sentiment["score"] > 0.75:
            tags.append(["L", f"nostr.event.content:nlp.sentiment:{sentiment['label'].lower()}"])

        if bool(re.search("http(s)?://", event["content"])):
            tags.append(["L", "nostr.event.content:has-links"])

        named_entities = self.__ner(event["content"])
        for entity_type in named_entities:
            if entity_type in self.ner_mapping.keys() and self.ner_mapping[entity_type]:
                tags.append(["L", f"nostr.event.content:nlp.ner.type:{self.ner_mapping[entity_type]}"])

        return tags

    def __ner(self, content):
        named_entities = self.ner_pipeline(content)
        confident_named_entities = list(
            filter(lambda n: n['score'] > 0.99, named_entities))
        return list(Counter(list(map(lambda x: x['entity'], confident_named_entities))))

    def __get_lang(self, content):
        language = self.fasttext_model.predict(re.sub("\n", " ", content))
        flanguage_code_flores_200 = re.sub("__label__", "", language[0][0])

        if not flanguage_code_flores_200 in self.iso_639_2_mapping.keys():
            return (None, 0)

        language_label = self.iso_639_2_mapping[flanguage_code_flores_200]
        language_score = language[1][0]
        return (language_label, language_score)

    def __content_length_to_range(self, content_length):
        if content_length <= 100:
            return "<=100"
        elif content_length <= 300:
            return "100-300"
        elif content_length <= 500:
            return "300-500"
        elif content_length <= 1000:
            return "500-1000"
        elif content_length <= 10000:
            return "1000-10000"
        else:
            return "10000+"

    def is_root(self, tags):
        lengthy_tags = list(filter(lambda c: len(c) > 3, tags))
        root_or_reply_tags = list(
            filter(lambda c: c[3] == 'root' or c[3] == 'reply', lengthy_tags))

        return len(root_or_reply_tags) == 0
