"""
orthography.py

Two independent post-processing concerns live here:

1. normalize_nahuatl_orthography() -- for the FINE-TUNED nhi model's
   output. VALIDATED FINDING: original_sentence (used as the training
   label) already follows the competition's phonemic convention, so the
   fine-tuned model's own output needs no correction. Stays a pass-through.
   Do NOT apply colonial_to_modern_nahuatl() below to fine-tuned output --
   it's already in the target convention; remapping it would corrupt it.

2. colonial_to_modern_nahuatl() -- for the STOCK MMS model's raw output
   (used for azz/nhw/nhi-via-stock/unmapped-fallback). VALIDATED FINDING:
   stock MMS decodes in colonial/SIL-style Spanish-influenced orthography
   (qu/c/hu/z/accents), while the competition's ground truth uses modern
   INALI/SEP convention (k/u/s, no accents). Confirmed via controlled
   25-clip-per-dialect local benchmarks against real ground truth:
       azz: 0.8603 -> 0.7095 WER  (+0.1508)
       nhw: 0.6997 -> 0.3944 WER  (+0.3053)
       nhi: 0.6691 -> 0.6240 WER  (+0.0451, via stock nhi adapter)
   Weighted-by-dev-share aggregate: ~0.7723 -> ~0.5806. This is now
   applied to every _run_inference() call that serves the STOCK model
   (see asr_model.py's apply_colonial_mapping parameter), not to the
   fine-tuned model's output.

Residual uncertainty worth knowing: colonial_to_modern_nahuatl() lowercases
its input. The competition's real WER normalizer's exact casing behavior
is only partially documented ("preserves mid-sentence casing"). These
benchmark numbers were measured with jiwer's own lowercasing applied to
BOTH prediction and reference locally, so they're valid for THAT
comparison -- but if the real scorer does NOT lowercase before comparing,
gains against proper nouns (e.g. "Target" vs "target") may be smaller in
the real score than measured here. Not a reason to withhold this change
(the underlying orthography fix is real and large), just a caveat on the
exact magnitude.
"""

import re
import unicodedata

SPANISH_KEEP_WORDS = {
    'actas',
    'acuerdo',
    'acuerdos',
    'aguacate',
    'aqui',
    'azufre',
    'balcon',
    'banco',
    'basico',
    'bueno',
    'caballo',
    'caballos',
    'cabana',
    'cable',
    'cachito',
    'cacho',
    'cada',
    'cadena',
    'cadenas',
    'caja',
    'calidra',
    'calificaciones',
    'cama',
    'cambio',
    'camino',
    'camiones',
    'camioneta',
    'campo',
    'canteadito',
    'capaz',
    'carcel',
    'carga',
    'cargas',
    'cargo',
    'carretera',
    'carro',
    'carros',
    'casa',
    'casi',
    'caso',
    'castigo',
    'catorce',
    'cerca',
    'cerveza',
    'cinco',
    'cincuenta',
    'coches',
    'cocina',
    'coctel',
    'col',
    'colado',
    'coleta',
    'colonia',
    'columna',
    'columnas',
    'combi',
    'comedor',
    'comida',
    'como',
    'companeros',
    'competente',
    'completo',
    'comunidad',
    'comunidades',
    'con',
    'conduciendo',
    'conecte',
    'confianza',
    'consejo',
    'construccion',
    'contrario',
    'copia',
    'coraje',
    'corral',
    'cosa',
    'cosas',
    'costeras',
    'costumbre',
    'credencial',
    'crema',
    'creo',
    'cuadrado',
    'cual',
    'cualquier',
    'cuando',
    'cuanto',
    'cuarenta',
    'cuarteaduras',
    'cuarto',
    'cuartos',
    'cuatro',
    'cubetas',
    'cuenta',
    'cuentos',
    'cuerpo',
    'cueva',
    'cuidado',
    'cultos',
    'cunado',
    'de',
    'delicada',
    'diez',
    'doce',
    'donde',
    'durazno',
    'duraznos',
    'educacion',
    'el',
    'embarazada',
    'escalera',
    'escuadra',
    'escuela',
    'escuelas',
    'esquina',
    'esquinas',
    'esquineros',
    'este',
    'exactamente',
    'exacto',
    'focos',
    'fuerza',
    'hasta',
    'hechuras',
    'hora',
    'huesos',
    'huevito',
    'huevo',
    'hule',
    'humberto',
    'humedad',
    'inquieto',
    'inspector',
    'juez',
    'juzgado',
    'la',
    'las',
    'lazo',
    'longaniza',
    'los',
    'loza',
    'lozas',
    'manzana',
    'maquina',
    'marzo',
    'mas',
    'menos',
    'mercancia',
    'mezcla',
    'micro',
    'miercoles',
    'muneco',
    'nunca',
    'paquete',
    'parque',
    'paz',
    'pedazo',
    'perfeccion',
    'pero',
    'pesos',
    'picante',
    'pico',
    'plastico',
    'poco',
    'porque',
    'prescolar',
    'que',
    'quesadillas',
    'quien',
    'quienes',
    'quiera',
    'quiero',
    'quince',
    'quinientos',
    'quizas',
    'razon',
    'recamaras',
    'recreo',
    'rectangular',
    'reforzado',
    'refresco',
    'refuerzo',
    'seca',
    'seccion',
    'secundaria',
    'siquiera',
    'tabique',
    'taco',
    'tampoco',
    'tanque',
    'taquitos',
    'tiempo',
    'unico',
    'veinticinco',
    'veinticuatro',
    'vez',
    'zapatas',
    'zapato'
}


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def colonial_to_modern_nahuatl(text: str) -> str:
    """
    Converts Classical/Spanish SIL orthography into Modern INALI/SEP
    orthography, while preserving standard Spanish loanwords. For STOCK
    MMS model output only -- see module docstring.
    """
    words = _strip_accents(text.lower()).split()
    processed_words = []

    for w in words:
        if w in SPANISH_KEEP_WORDS:
            processed_words.append(w)
            continue

        # 1. 'hu' -> 'u' (e.g., huan -> uan, huelis -> uelis)
        w = re.sub(r"hu", "u", w)

        # 2. 'qu' -> 'k' (e.g., quena -> kena, sequin -> sekin)
        w = re.sub(r"qu", "k", w)

        # 3. 'c' -> 'k' before a, o, u, consonants, or word-ends (preserves 'ch' and 'ce/ci')
        w = re.sub(r"c(?=[aou]|$|[^heiaou])", "k", w)

        # 4. 'z' -> 's' (e.g., amantzil -> amantsil)
        w = re.sub(r"z", "s", w)

        processed_words.append(w)

    return " ".join(processed_words)


def normalize_nahuatl_orthography(text: str) -> str:
    # Deliberate no-op for the FINE-TUNED model's output -- see module
    # docstring point 1. Do not apply colonial_to_modern_nahuatl() here.
    return text