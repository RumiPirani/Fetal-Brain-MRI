"""Differential diagnosis engine from SPEC 4.6 and 6.5."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fetal_brain_mri.calculator import ParameterResult
    from fetal_brain_mri.inputs import MeasurementInput

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_MILD_VM_LOW_MM = 10.0
_MILD_VM_HIGH_MM = 15.0
_SEVERE_VM_MM = 15.0
_VM_ASYMMETRY_MM = 2.0
_CSP_ABSENT_MM = 1.0
_CSP_ENLARGED_MM = 10.0
_THIRD_V_WIDE_MM = 3.5
_ABNORMAL_Z_LOW = -1.645   # 5th percentile
_ABNORMAL_Z_HIGH = 1.645   # 95th percentile
_MICRO_MACRO_Z = 2.0       # 3rd/97th percentile boundary
_CHIARI_Z_THRESHOLD = -2.0
_CHIARI_ONTD_POSTERIOR_THRESHOLD = 0.5
_DWM_TVA_DEGREES = 35.0    # Whitehead 2022 refined threshold


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DdxDiagnosis:
    """One row in a DDx card's likelihood table."""

    name: str
    likelihood: str   # qualitative or quantitative string, per SPEC §7.4
    rationale: str


@dataclass(frozen=True)
class DdxCard:
    """A fired differential-diagnosis card."""

    card_id: str
    label: str
    trigger_description: str
    diagnoses: tuple[DdxDiagnosis, ...]
    next_steps: str
    limitations: str
    primary_source: str
    secondary_source: str
    is_combined_pattern: bool = False


# ---------------------------------------------------------------------------
# Card definitions (verbatim clinical content from SPEC §4.6 and §7.4)
# ---------------------------------------------------------------------------

def _mild_vm_card(trigger: str) -> DdxCard:
    return DdxCard(
        card_id="mild_ventriculomegaly",
        label="Mild-to-moderate ventriculomegaly",
        trigger_description=trigger,
        diagnoses=(
            DdxDiagnosis("Isolated / idiopathic", "~70-80% of cases", "Most common; ND delay ~7.9% (Pagani 2014)."),
            DdxDiagnosis("Chromosomal abnormality (e.g. T21)", "~5%", "Aneuploidy is a significant cause (SMFM 2018)."),
            DdxDiagnosis("Agenesis of the corpus callosum", "~5%", "Frequently associated with VM (SMFM 2018)."),
            DdxDiagnosis("Aqueductal stenosis", "~5-10%", "Common cause of obstructive VM."),
            DdxDiagnosis("Congenital infection (CMV)", "~2-5%", "Important to exclude (SMFM 2018)."),
        ),
        next_steps=(
            "Recommend dedicated views of corpus callosum, fetal MRI follow-up at 32 weeks, "
            "and TORCH/CMV screening."
        ),
        limitations=(
            "Likelihoods derived from cohort studies; actual risk depends on additional findings, "
            "karyotype, and CMV status."
        ),
        primary_source="Pagani G et al. Ultrasound Obstet Gynecol. 2014;44(3):254-260.",
        secondary_source="SMFM. Mild fetal ventriculomegaly. Am J Obstet Gynecol. 2018;219(1):B2-B9.",
    )


def _severe_vm_card(trigger: str) -> DdxCard:
    return DdxCard(
        card_id="severe_ventriculomegaly",
        label="Severe ventriculomegaly (>=15 mm)",
        trigger_description=trigger,
        diagnoses=(
            DdxDiagnosis("Aqueductal stenosis", "~20%", "Most common cause of obstructive hydrocephalus."),
            DdxDiagnosis("Associated CNS / non-CNS anomaly", "High", "Frequently associated; worsens prognosis."),
            DdxDiagnosis("Chromosomal abnormality", "Significant", "Risk increases with severity."),
            DdxDiagnosis("Congenital infection (CMV / toxoplasmosis)", "~1-5%", "Known cause (Giorgione 2022)."),
            DdxDiagnosis("Isolated / idiopathic", "~10-20%", "Diagnosis of exclusion after extensive workup."),
        ),
        next_steps=(
            "Recommend detailed neurosonography and fetal MRI, invasive genetic testing with chromosomal "
            "microarray, and screening for congenital infections. Multidisciplinary fetal neurology consultation."
        ),
        limitations=(
            "Likelihoods are estimates; actual risk depends on associated anomalies, karyotype, and infection status."
        ),
        primary_source="Giorgione V et al. Prenat Diagn. 2022;42(13):1674-1681.",
        secondary_source="Carta S et al. Ultrasound Obstet Gynecol. 2018;52(2):165-173.",
    )


def _asymmetric_vm_card(delta_mm: float) -> DdxCard:
    return DdxCard(
        card_id="asymmetric_ventricles",
        label="Asymmetric lateral ventricles",
        trigger_description=f"Side-to-side atrial difference {delta_mm:.1f} mm > 2 mm threshold.",
        diagnoses=(
            DdxDiagnosis("Isolated / benign variant", "High", "VABS-II scores within normal range for isolated asymmetry (Meyer 2018)."),
            DdxDiagnosis("Progression to ventriculomegaly", "~37-46%", "Systematic review (Sgayer 2025)."),
            DdxDiagnosis("Associated CNS anomalies", "~24%", "24.2% in asymmetric VM cases (Barzilay 2017)."),
            DdxDiagnosis("Genetic abnormalities", "<5%", "Association reported (Sgayer 2025)."),
            DdxDiagnosis("Intrauterine infection (CMV)", "<5%", "Known cause of fetal brain anomalies."),
        ),
        next_steps=(
            "Recommend detailed fetal neurosonogram and fetal MRI to rule out associated anomalies. "
            "Consider genetic counselling and TORCH screening."
        ),
        limitations="Likelihoods are estimates; actual risk depends on associated findings and genetic testing.",
        primary_source="Barzilay E et al. AJNR. 2017;38(2):371-375.",
        secondary_source="Meyer R et al. Ultrasound Obstet Gynecol. 2018;52(4):467-472.",
    )


def _small_tcd_card() -> DdxCard:
    return DdxCard(
        card_id="small_tcd",
        label="Small transcerebellar diameter (<5th percentile)",
        trigger_description="TCD consensus z-score below -1.645.",
        diagnoses=(
            DdxDiagnosis("Chromosomal abnormalities (aneuploidy, CNVs)", "~55%", "Pathogenic CNVs in 54.6% of cerebellar hypoplasia (Zou 2018)."),
            DdxDiagnosis("Genetic syndromes (Joubert, CHARGE)", "~15-25%", "Numerous syndromes associated (Aldinger 2016)."),
            DdxDiagnosis("Associated CNS / non-CNS anomalies", "~10-20%", "Often seen with structural anomalies (Howley 2018)."),
            DdxDiagnosis("Congenital infection (CMV, Zika)", "~5-15%", "Infections cause cerebellar disruption (Howley 2018)."),
            DdxDiagnosis("Isolated / idiopathic", "~10-20%", "Significant portion without clear etiology."),
        ),
        next_steps=(
            "Recommend detailed neurosonography, fetal MRI for associated anomalies, amniocentesis "
            "with chromosomal microarray, and TORCH screening. Genetic counselling advised."
        ),
        limitations="Likelihoods vary based on associated findings and specific genetic testing.",
        primary_source="Zou Z et al. Prenat Diagn. 2018;38(2):91-98.",
        secondary_source="Aldinger KA, Doherty D. Semin Fetal Neonatal Med. 2016;21(5):321-332.",
    )


def _large_tcd_card() -> DdxCard:
    return DdxCard(
        card_id="large_tcd",
        label="Large transcerebellar diameter (>95th percentile)",
        trigger_description="TCD consensus z-score above +1.645.",
        diagnoses=(
            DdxDiagnosis("Overgrowth syndrome (Sotos, BWS)", "Variable", "Large cerebellum is a recognised feature of overgrowth syndromes."),
            DdxDiagnosis("Isolated macrocerebellum / normal variant", "Most cases", "Often incidental."),
            DdxDiagnosis("Megalencephaly", "Variable", "Disorder of neuronal proliferation."),
        ),
        next_steps=(
            "Recommend detailed fetal neurosonography and MRI. Genetic counselling if associated anomalies present."
        ),
        limitations="Limited literature on isolated macrocerebellum; likelihoods are qualitative.",
        primary_source="Bosemani T et al. RadioGraphics. 2015;35(1):200-220.",
        secondary_source="Bosemani T et al. RadioGraphics. 2015;35(1):200-220.",
    )


def _vermian_hypoplasia_card() -> DdxCard:
    return DdxCard(
        card_id="vermian_hypoplasia",
        label="Vermian hypoplasia (<5th percentile)",
        trigger_description="Vermis CC or AP diameter z-score below -1.645.",
        diagnoses=(
            DdxDiagnosis("Isolated vermian hypoplasia", "~20-30%", "Variable prognosis (Poretti 2014)."),
            DdxDiagnosis("Dandy-Walker malformation", "~30-40%", "Classic triad: vermian agenesis, cystic 4th V, enlarged PF (SMFM 2020)."),
            DdxDiagnosis("Chromosomal abnormality (T18, T13)", "~15-20%", "Associated with DWM in ~16% (SMFM 2020)."),
            DdxDiagnosis("Blake's pouch cyst", "~5-10%", "More favourable prognosis; vermis rotated superiorly."),
            DdxDiagnosis("Genetic syndromes (Joubert, CHARGE)", "~5%", "Require specific genetic testing (Poretti 2014)."),
            DdxDiagnosis("Prenatal infection (CMV)", "<5%", "CMV can disrupt cerebellar development."),
        ),
        next_steps=(
            "Recommend detailed multiplanar neurosonography and fetal MRI. Offer genetic counselling "
            "with chromosomal microarray and TORCH screening."
        ),
        limitations="Likelihoods depend on full constellation of findings, GA, and genetic testing.",
        primary_source="Poretti A et al. Am J Med Genet C. 2014;166C(2):211-226.",
        secondary_source="SMFM; Monteagudo A. Dandy-Walker Malformation. Am J Obstet Gynecol. 2020;223(6):B38-B41.",
    )


def _small_pons_card() -> DdxCard:
    return DdxCard(
        card_id="small_pons",
        label="Small pons AP diameter (<5th percentile)",
        trigger_description="Pons AP z-score below -1.645.",
        diagnoses=(
            DdxDiagnosis("Pontocerebellar hypoplasia Type 2 (PCH2A)", "~40-50%", "Most common PCH type (van Dijk 2018)."),
            DdxDiagnosis("Pontocerebellar hypoplasia Type 1", "~10-20%", "Second most common (van Dijk 2018)."),
            DdxDiagnosis("Other PCH types (3, 4, 5, 6…)", "~10%", "Numerous other genetic subtypes (van Dijk 2018)."),
            DdxDiagnosis("CASK-related disorders", "~5-10%", "Can present with PCH (Moog 2020)."),
            DdxDiagnosis("Tubulinopathies", "~5%", "Associated with brain malformations (Bahi-Buisson 2021)."),
        ),
        next_steps=(
            "Recommend targeted gene panel for PCH and brain malformation genes (TSEN54, CASK, tubulinopathies). "
            "Fetal MRI for associated anomalies. Genetic counselling."
        ),
        limitations="Likelihoods are estimates for rare diseases; definitive diagnosis requires genetic testing.",
        primary_source="van Dijk T et al. Orphanet J Rare Dis. 2018;13:92.",
        secondary_source="Sánchez-Albisua I et al. Orphanet J Rare Dis. 2014;9:70.",
    )


def _csp_absent_card() -> DdxCard:
    return DdxCard(
        card_id="csp_absent",
        label="Absent cavum septum pellucidum",
        trigger_description="CSP reported as absent or measured < 1 mm.",
        diagnoses=(
            DdxDiagnosis("Holoprosencephaly (HPE)", "~50-60%", "Most common association, esp. with facial anomalies (Malinger 2005)."),
            DdxDiagnosis("Agenesis of the corpus callosum (ACC)", "~55%", "Absent CSP in ~2/3 of ACC cases (SMFM 2020)."),
            DdxDiagnosis("Severe hydrocephalus / VM", "~10-20%", "Increased ventricular pressure can destroy septal leaves (Malinger 2005)."),
            DdxDiagnosis("Septo-optic dysplasia (SOD)", "~5-10%", "Classic association; optic nerve hypoplasia difficult to confirm prenatally."),
            DdxDiagnosis("Schizencephaly", "<5%", "Rare but important cause."),
            DdxDiagnosis("Isolated / idiopathic", "<5%", "Diagnosis of exclusion (Malinger 2005)."),
        ),
        next_steps=(
            "Recommend dedicated multiplanar neurosonography, coronal and sagittal views for CC, "
            "forebrain, and optic nerves. Fetal MRI strongly recommended."
        ),
        limitations="Final diagnosis depends on detailed fetal neuroimaging, genetic testing, and exclusion of TORCH infections.",
        primary_source="Malinger G et al. Ultrasound Obstet Gynecol. 2005;25(1):42-49.",
        secondary_source="SMFM; Ward A, Monteagudo A. Absent CSP. Am J Obstet Gynecol. 2020;223(6):B23-B26.",
    )


def _csp_enlarged_card() -> DdxCard:
    return DdxCard(
        card_id="csp_enlarged",
        label="Enlarged cavum septum pellucidum (>10 mm)",
        trigger_description="CSP width > 10 mm.",
        diagnoses=(
            DdxDiagnosis("Normal variant / isolated finding", "~85-90%", "High rate; neurodevelopmental delay still possible (Ding 2019)."),
            DdxDiagnosis("Cavum vergae", "~5-10%", "Common posterior extension of the CSP."),
            DdxDiagnosis("Cavum velum interpositum cyst", "<5%", "Triangular shape, more posterior."),
            DdxDiagnosis("Associated CNS / non-CNS anomalies", "~1-5%", "Altered prognosis with associated findings."),
            DdxDiagnosis("Symptomatic / obstructive hydrocephalus", "<1%", "Rarely a large CSP cyst causes obstruction."),
        ),
        next_steps=(
            "Recommend detailed fetal neurosonogram and fetal MRI to confirm finding, rule out associated "
            "anomalies, and differentiate from other midline cystic structures."
        ),
        limitations="Likelihoods are estimates; detailed MRI and neurosonogram recommended for accurate diagnosis.",
        primary_source="Ding H et al. Eur J Obstet Gynecol Reprod Biol. 2019;237:85-88.",
        secondary_source="Nunes JS et al. J Med Ultrasound. 2024;33(3):289-290.",
    )


def _cc_absent_card() -> DdxCard:
    return DdxCard(
        card_id="cc_absent",
        label="Absent corpus callosum",
        trigger_description="CC reported as absent.",
        diagnoses=(
            DdxDiagnosis("Isolated / idiopathic complete ACC", "~65-75% normal ND", "Normal neurodevelopment in 65-75% of isolated ACC (Santo 2012)."),
            DdxDiagnosis("Monogenic syndrome", "~30%", "Monogenic disorders identified in 30% of one cohort (Sun 2024)."),
            DdxDiagnosis("Chromosomal abnormality / CNV", "~15-20%", "Overall rate ~18% (Santo 2012)."),
            DdxDiagnosis("Associated CNS malformations", "Varies", "ACC frequently with hydrocephalus or cerebellar dysplasia."),
            DdxDiagnosis("In utero insult (ischemic / infectious)", "Unknown", "Secondary dysgenesis from destructive events."),
        ),
        next_steps=(
            "Recommend detailed fetal neurosonography and MRI to confirm and search for associated anomalies. "
            "Offer invasive genetic testing with chromosomal microarray and consider whole-exome sequencing. "
            "TORCH/CMV screening and genetic counselling."
        ),
        limitations="Prognosis depends heavily on associated anomalies; requires comprehensive imaging and genetic evaluation.",
        primary_source="Sun H et al. Eur J Obstet Gynecol Reprod Biol. 2024;298:146-152.",
        secondary_source="Santo S et al. Ultrasound Obstet Gynecol. 2012;40(5):513-521.",
    )


def _cc_short_card() -> DdxCard:
    return DdxCard(
        card_id="cc_short",
        label="Short / dysgenetic corpus callosum (<5th percentile)",
        trigger_description="CC length z-score below -1.645 with CSP present.",
        diagnoses=(
            DdxDiagnosis("Partial / hypogenetic ACC", "Most cases", "CC present but abnormally short."),
            DdxDiagnosis("Monogenic syndrome", "~25-30%", "Consistent with Sun 2024 cohort, same caveat."),
            DdxDiagnosis("Chromosomal / CNV", "~15%", "Estimate."),
        ),
        next_steps=(
            "Recommend postnatal MRI for confirmation. Genetic counselling with chromosomal microarray."
        ),
        limitations="Likelihoods are estimates.",
        primary_source="Garel C et al. AJNR. 2011;32(8):1436-1443.",
        secondary_source="Santo S et al. Ultrasound Obstet Gynecol. 2012;40(5):513-521.",
    )


def _third_ventricle_wide_card() -> DdxCard:
    return DdxCard(
        card_id="third_ventricle_dilatation",
        label="Dilated third ventricle (>3.5 mm)",
        trigger_description="Third ventricle width > 3.5 mm (Hertzberg 1997 threshold).",
        diagnoses=(
            DdxDiagnosis("Aqueductal stenosis", "~55%", "Common cause of obstructive hydrocephalus (Hertzberg 1997)."),
            DdxDiagnosis("Agenesis / dysgenesis of CC", "~10-20%", "CC malformation can alter CSF dynamics."),
            DdxDiagnosis("Holoprosencephaly (mild / lobar)", "~5-15%", "Incomplete forebrain cleavage."),
            DdxDiagnosis("Interhemispheric cyst / cavum veli interpositi", "~5-10%", "Mass effect near 3rd ventricle."),
            DdxDiagnosis("Isolated finding", "Variable", "May be normal variant; close follow-up warranted."),
        ),
        next_steps=(
            "Recommend detailed fetal neurosonography and fetal MRI to assess for associated anomalies, "
            "particularly of CC and aqueduct. Karyotype and TORCH screening. Serial follow-up."
        ),
        limitations=(
            "Third-ventricle z-scores from a cross-modality approximation (Birnbaum 2018 — 3D US source). "
            "Treat z-score as ordinal; the 3.5 mm threshold from Hertzberg 1997 is the primary decision criterion."
        ),
        primary_source="Hertzberg BS et al. Radiology. 1997;203(3):641-644.",
        secondary_source="Giorgione V et al. Prenat Diagn. 2022;42(13):1674-1681.",
    )


def _microcephaly_card() -> DdxCard:
    return DdxCard(
        card_id="microcephaly_pattern",
        label="Microcephaly (skull BPD <3rd percentile)",
        trigger_description="Skull BPD z-score <= -1.881 (<3rd percentile).",
        diagnoses=(
            DdxDiagnosis("Genetic etiologies (syndromic / non-syndromic)", "~50%", "Genetic factors in ~50% (Wang 2023)."),
            DdxDiagnosis("Congenital infections (CMV, Zika, Toxo)", "~15-20%", "Major cause of acquired microcephaly (Hanzlik 2017)."),
            DdxDiagnosis("Perinatal brain injury (hypoxic / ischemic)", "~15-20%", "Insults disrupt brain development (von der Hagen 2014)."),
            DdxDiagnosis("Maternal / environmental factors", "~5-10%", "Teratogens, malnutrition, metabolic disease (ISUOG 2019)."),
            DdxDiagnosis("Isolated / idiopathic", "Variable", "Significant portion without identifiable cause."),
        ),
        next_steps=(
            "Recommend detailed fetal neurosonogram and fetal MRI. Genetic counselling with chromosomal "
            "microarray and / or exome sequencing. Maternal TORCH testing."
        ),
        limitations="Likelihoods are estimates; actual risk depends on severity, associated anomalies, and family history.",
        primary_source="Wang Y et al. Front Genet. 2023;14:1112153.",
        secondary_source="Hanzlik E, Gigante J. Children (Basel). 2017;4(6):47.",
    )


def _macrocephaly_card() -> DdxCard:
    return DdxCard(
        card_id="macrocephaly_pattern",
        label="Macrocephaly (skull BPD >97th percentile)",
        trigger_description="Skull BPD z-score >= +1.881 (>97th percentile).",
        diagnoses=(
            DdxDiagnosis("Benign familial macrocephaly", "~50-60%", "Most common cause; family history of large head (Shinar 2023)."),
            DdxDiagnosis("Megalencephaly (non-syndromic)", "Variable", "Disorder of neuronal proliferation (Shinar 2023)."),
            DdxDiagnosis("Hydrocephalus", "~10-20%", "Common secondary cause."),
            DdxDiagnosis("Genetic syndromes (Sotos, PTEN)", "~5-10%", "Overgrowth syndromes."),
            DdxDiagnosis("Brain tumour", "<1%", "Rare cause."),
        ),
        next_steps=(
            "Recommend detailed fetal neurosonography and fetal MRI. Genetic counselling and chromosomal "
            "microarray especially if other anomalies present. Detailed family head-size history."
        ),
        limitations="Likelihoods are estimates; actual risk depends on imaging, family history, and genetic testing.",
        primary_source="Shinar S et al. Prenat Diagn. 2023;43(13):1650-1661.",
        secondary_source="The Fetal Medicine Foundation. Macrocephaly.",
    )


def _acc_combined_card() -> DdxCard:
    return DdxCard(
        card_id="acc_pattern",
        label="Agenesis of corpus callosum (combined pattern)",
        trigger_description="CC absent and CSP absent together.",
        diagnoses=(
            DdxDiagnosis("Complete ACC — isolated", "~65-75% normal ND", "Normal neurodevelopment in 65-75% of isolated ACC (Santo 2012)."),
            DdxDiagnosis("Monogenic syndrome", "~30%", "Sun 2024 cohort (n=40); exome sequencing indicated."),
            DdxDiagnosis("Chromosomal / pathogenic CNV", "~15-20%", "Overall rate ~18%."),
        ),
        next_steps=(
            "Complete ACC confirmed by absent CSP and CC. Recommend detailed fetal MRI, chromosomal "
            "microarray, whole-exome sequencing, TORCH/CMV screening, and genetic counselling."
        ),
        limitations="Prognosis depends entirely on associated anomalies; requires comprehensive evaluation.",
        primary_source="Santo S et al. Ultrasound Obstet Gynecol. 2012;40(5):513-521.",
        secondary_source="Sun H et al. Eur J Obstet Gynecol Reprod Biol. 2024;298:146-152.",
        is_combined_pattern=True,
    )


def _hpe_combined_card() -> DdxCard:
    return DdxCard(
        card_id="hpe_pattern",
        label="Holoprosencephaly (combined pattern)",
        trigger_description="Severe ventriculomegaly with absent CSP — pattern consistent with HPE.",
        diagnoses=(
            DdxDiagnosis("Alobar / semilobar HPE", "~70%", "Most severe HPE forms (Malinger 2005 / SMFM 2020)."),
            DdxDiagnosis("Lobar HPE", "~15%", "Mildest form; midline structures partially preserved."),
            DdxDiagnosis("Septo-optic dysplasia (SOD)", "~5%", "Absent CSP with optic nerve hypoplasia."),
        ),
        next_steps=(
            "Fetal MRI for characterisation of HPE subtype and associated anomalies. "
            "Chromosomal microarray and exome sequencing. Genetic counselling regarding very poor prognosis for alobar HPE."
        ),
        limitations="Likelihoods are estimates; definitive HPE diagnosis requires midline anatomy assessment on MRI.",
        primary_source="Malinger G et al. Ultrasound Obstet Gynecol. 2005;25(1):42-49.",
        secondary_source="SMFM; Ward A, Monteagudo A. Absent CSP. Am J Obstet Gynecol. 2020;223(6):B23-B26.",
        is_combined_pattern=True,
    )


def _hydrocephalus_combined_card() -> DdxCard:
    return DdxCard(
        card_id="hydrocephalus_pattern",
        label="Obstructive hydrocephalus (combined pattern)",
        trigger_description="Severe ventriculomegaly with dilated third ventricle (>3.5 mm).",
        diagnoses=(
            DdxDiagnosis("Aqueductal stenosis", "~70%", "Most common cause of triventricular hydrocephalus."),
            DdxDiagnosis("X-linked L1CAM mutation", "~5-10%", "Causes X-linked aqueductal stenosis."),
            DdxDiagnosis("Posterior fossa mass / Chiari II", "~10%", "Causes obstructive hydrocephalus."),
        ),
        next_steps=(
            "Dedicated evaluation of the aqueduct on fetal MRI sagittal sequences. "
            "Genetic testing including L1CAM if severe bilateral VM with dilated 3rd ventricle."
        ),
        limitations="Likelihoods are estimates.",
        primary_source="Heaphy-Henault KJ et al. AJNR. 2018;39(5):942-948.",
        secondary_source="Hertzberg BS et al. Radiology. 1997;203(3):641-644.",
        is_combined_pattern=True,
    )


def _dwm_combined_card() -> DdxCard:
    return DdxCard(
        card_id="dandy_walker_spectrum",
        label="Dandy-Walker spectrum (combined pattern)",
        trigger_description="Vermian hypoplasia with tegmento-vermian angle >35°.",
        diagnoses=(
            DdxDiagnosis("Dandy-Walker malformation", "~60%", "Classic triad: vermian agenesis, cystic 4th V, enlarged PF (Whitehead 2022)."),
            DdxDiagnosis("Vermian hypoplasia variant", "~25%", "Small vermis without full DWM criteria."),
            DdxDiagnosis("Blake's pouch remnant", "~10%", "Mildly raised TVA with normal-sized vermis."),
        ),
        next_steps=(
            "Recommend detailed multiplanar fetal MRI. Chromosomal microarray and exome sequencing. "
            "TORCH screening. Genetic counselling regarding variable prognosis."
        ),
        limitations="DWM vs Blake's pouch vs vermian hypoplasia distinction requires careful TVA and vermis morphology assessment.",
        primary_source="Whitehead MT et al. AJNR. 2022;43(10):1488-1493.",
        secondary_source="SMFM; Monteagudo A. Dandy-Walker Malformation. Am J Obstet Gynecol. 2020;223(6):B38-B41.",
        is_combined_pattern=True,
    )


def _pch_combined_card() -> DdxCard:
    return DdxCard(
        card_id="pch_pattern",
        label="Pontocerebellar hypoplasia (combined pattern)",
        trigger_description="Small pons and small TCD — pattern consistent with PCH.",
        diagnoses=(
            DdxDiagnosis("PCH Type 2", "~50%", "Most common PCH type (van Dijk 2018)."),
            DdxDiagnosis("PCH Type 1", "~15%", "Motor neuronopathy."),
            DdxDiagnosis("Other PCH / CASK / tubulinopathy", "~20%", "Numerous genetic subtypes."),
            DdxDiagnosis("Acquired (CMV)", "~5%", "Infection-induced posterior fossa disruption."),
        ),
        next_steps=(
            "Targeted gene panel for PCH genes (TSEN54, CASK, tubulinopathies). "
            "Fetal MRI for cerebral and cerebellar anomaly assessment."
        ),
        limitations="Rare disease; likelihoods are estimates from literature.",
        primary_source="van Dijk T et al. Orphanet J Rare Dis. 2018;13:92.",
        secondary_source="van Dijk T et al. Orphanet J Rare Dis. 2018;13:92.",
        is_combined_pattern=True,
    )


def _chiari_ii_card(p_ontd: float) -> DdxCard:
    return DdxCard(
        card_id="chiari_ii_open_ntd",
        label="Chiari II / open neural tube defect (combined pattern)",
        trigger_description=(
            f"TDPF z < -2 and CSA z < -2; Mahalanobis ONTD posterior probability {p_ontd:.0%}."
        ),
        diagnoses=(
            DdxDiagnosis(
                "Chiari II — open neural tube defect (myelomeningocele / myeloschisis)",
                "~85-90% when both z-scores below -2",
                "~91% sensitivity and ~93% specificity for ONTD vs controls (Woitek 2014).",
            ),
            DdxDiagnosis("Closed neural tube defect (lipomyelomeningocele, meningocele)", "~5-10%", "Milder posterior-fossa changes than open NTDs (Woitek 2014)."),
            DdxDiagnosis("Severe vermian hypoplasia / Dandy-Walker spectrum", "~3-5%", "CSA typically preserved or increased in DWM."),
            DdxDiagnosis("Benign small posterior fossa", "<1%", "Rarely both z-scores fall below -2 in a healthy fetus."),
        ),
        next_steps=(
            "Recommend dedicated examination of the fetal spine on the same MRI examination (sagittal and axial T2) "
            "to identify the open NTD lesion and document its level. Refer to a fetal-surgery-capable centre. "
            "Genetic counselling with chromosomal microarray. Review maternal alpha-fetoprotein."
        ),
        limitations=(
            "Discriminator trained on a single-centre cohort (n=44 Woitek 2014) and validated on 60 open NTD cases "
            "(Aertsen 2019). Performance uncertain outside GA 21-37 weeks or with motion-degraded mid-sagittal images. "
            "FLAG: research-mode card pending local calibration (SPEC §7.5)."
        ),
        primary_source="Woitek R et al. PLOS ONE. 2014;9(11):e112585.",
        secondary_source="Aertsen M et al. AJNR. 2019;40(1):191-198.",
        is_combined_pattern=True,
    )


# ---------------------------------------------------------------------------
# Mahalanobis discriminator for Chiari II (SPEC §6.5.3)
# ---------------------------------------------------------------------------

def _chiari_mahalanobis_ontd_posterior(z_tdpf: float, z_csa: float) -> float:
    """Return posterior probability of ONTD group using Woitek 2014 centroids."""

    # Group centroids and covariances in z-score space
    # Controls: mu=(0,0), sigma=(1,1), no covariance
    # ONTD: mu=(-3.6,-2.6), sigma=(0.9,1.1), correlation r≈0.5
    # CNTD: mu=(-1.4,-0.6), sigma=(1,1), no covariance (approx)

    z = (z_tdpf, z_csa)

    def _mahal_diagonal(z_vec: tuple[float, float], mu: tuple[float, float], sigma: tuple[float, float]) -> float:
        return ((z_vec[0] - mu[0]) / sigma[0]) ** 2 + ((z_vec[1] - mu[1]) / sigma[1]) ** 2

    def _mahal_ontd(z_vec: tuple[float, float]) -> float:
        # Full 2x2 covariance with r=0.5 between TDPF and CSA z-scores
        mu = (-3.6, -2.6)
        s1, s2, r = 0.9, 1.1, 0.5
        dz0 = z_vec[0] - mu[0]
        dz1 = z_vec[1] - mu[1]
        denom = 1 - r**2
        return (dz0**2 / s1**2 - 2 * r * dz0 * dz1 / (s1 * s2) + dz1**2 / s2**2) / denom

    d2_controls = _mahal_diagonal(z, (0.0, 0.0), (1.0, 1.0))
    d2_ontd = _mahal_ontd(z)
    d2_cntd = _mahal_diagonal(z, (-1.4, -0.6), (1.0, 1.0))

    # Posterior proportional to exp(-D²/2)
    scores = [math.exp(-d2_controls / 2), math.exp(-d2_ontd / 2), math.exp(-d2_cntd / 2)]
    total = sum(scores)
    if total == 0:
        return 0.0
    return scores[1] / total  # ONTD posterior


# ---------------------------------------------------------------------------
# Main DDx evaluation function
# ---------------------------------------------------------------------------

def evaluate_ddx(
    params: dict[str, "ParameterResult"],
    inputs: "MeasurementInput",
) -> list[DdxCard]:
    """Evaluate all DDx triggers and return fired cards in clinical order.

    Base trigger cards are evaluated first; combined-pattern cards that
    subsume base cards are appended after.
    """

    fired: list[DdxCard] = []
    combined: list[DdxCard] = []

    # Helper: consensus z for a parameter if evaluated
    def z(pid: str) -> float | None:
        r = params.get(pid)
        return r.consensus.consensus_z if r is not None else None

    def mm(pid: str) -> float | None:
        r = params.get(pid)
        return r.measurement if r is not None else None

    # --- Atrial measurements -----------------------------------------------
    atr_r = mm("atrium_r")
    atr_l = mm("atrium_l")

    mild_r = atr_r is not None and _MILD_VM_LOW_MM <= atr_r < _MILD_VM_HIGH_MM
    mild_l = atr_l is not None and _MILD_VM_LOW_MM <= atr_l < _MILD_VM_HIGH_MM
    severe_r = atr_r is not None and atr_r >= _SEVERE_VM_MM
    severe_l = atr_l is not None and atr_l >= _SEVERE_VM_MM

    if mild_r or mild_l:
        sides = []
        if mild_r:
            sides.append(f"right {atr_r:.1f} mm")
        if mild_l:
            sides.append(f"left {atr_l:.1f} mm")
        fired.append(_mild_vm_card(f"Atrial diameter: {', '.join(sides)}."))

    if severe_r or severe_l:
        sides = []
        if severe_r:
            sides.append(f"right {atr_r:.1f} mm")
        if severe_l:
            sides.append(f"left {atr_l:.1f} mm")
        fired.append(_severe_vm_card(f"Atrial diameter >=15 mm: {', '.join(sides)}."))

    if atr_r is not None and atr_l is not None:
        delta = abs(atr_r - atr_l)
        if delta > _VM_ASYMMETRY_MM:
            fired.append(_asymmetric_vm_card(delta))

    # --- TCD ---------------------------------------------------------------
    z_tcd = z("tcd")
    if z_tcd is not None:
        if z_tcd < -_ABNORMAL_Z_HIGH:
            fired.append(_small_tcd_card())
        elif z_tcd > _ABNORMAL_Z_HIGH:
            fired.append(_large_tcd_card())

    # --- Vermis ------------------------------------------------------------
    z_vcc = z("vermis_cc")
    z_vap = z("vermis_ap")
    vermian_low = (z_vcc is not None and z_vcc < -_ABNORMAL_Z_HIGH) or (
        z_vap is not None and z_vap < -_ABNORMAL_Z_HIGH
    )
    if vermian_low:
        fired.append(_vermian_hypoplasia_card())

    # --- Pons --------------------------------------------------------------
    z_pons = z("pons_ap")
    pons_low = z_pons is not None and z_pons < -_ABNORMAL_Z_HIGH
    if pons_low:
        fired.append(_small_pons_card())

    # --- CSP ---------------------------------------------------------------
    csp_mm = mm("csp")
    csp_absent = inputs.csp_absent or (csp_mm is not None and csp_mm < _CSP_ABSENT_MM)
    if csp_absent:
        fired.append(_csp_absent_card())
    elif csp_mm is not None and csp_mm > _CSP_ENLARGED_MM:
        fired.append(_csp_enlarged_card())

    # --- Corpus callosum ---------------------------------------------------
    cc_absent = inputs.cc_absent
    z_cc = z("cc_length")
    if cc_absent:
        fired.append(_cc_absent_card())
    elif z_cc is not None and z_cc < -_ABNORMAL_Z_HIGH:
        fired.append(_cc_short_card())

    # --- Third ventricle ---------------------------------------------------
    tv_mm = mm("third_ventricle")
    third_v_wide = tv_mm is not None and tv_mm > _THIRD_V_WIDE_MM
    if third_v_wide:
        fired.append(_third_ventricle_wide_card())

    # --- Skull size --------------------------------------------------------
    z_sbpd = z("skull_bpd")
    if z_sbpd is not None:
        if z_sbpd <= -_MICRO_MACRO_Z:
            fired.append(_microcephaly_card())
        elif z_sbpd >= _MICRO_MACRO_Z:
            fired.append(_macrocephaly_card())

    # --- Chiari II (SPEC §6.5) --------------------------------------------
    z_tdpf = z("tdpf")
    z_csa = z("csa")
    if (
        z_tdpf is not None
        and z_csa is not None
        and z_tdpf < _CHIARI_Z_THRESHOLD
        and z_csa < _CHIARI_Z_THRESHOLD
    ):
        p_ontd = _chiari_mahalanobis_ontd_posterior(z_tdpf, z_csa)
        if p_ontd > _CHIARI_ONTD_POSTERIOR_THRESHOLD:
            combined.append(_chiari_ii_card(p_ontd))

    # --- Combined patterns -------------------------------------------------
    # ACC: absent CC + absent CSP
    if cc_absent and csp_absent:
        combined.append(_acc_combined_card())

    # HPE: severe VM + absent CSP (without ACC — when CC not confirmed absent)
    if (severe_r or severe_l) and csp_absent and not cc_absent:
        combined.append(_hpe_combined_card())

    # Hydrocephalus: severe VM + wide third ventricle
    if (severe_r or severe_l) and third_v_wide:
        combined.append(_hydrocephalus_combined_card())

    # DWM: vermian hypoplasia + TVA above threshold
    if vermian_low and inputs.tva_degrees is not None and inputs.tva_degrees > _DWM_TVA_DEGREES:
        combined.append(_dwm_combined_card())

    # PCH: small pons + small TCD
    if pons_low and z_tcd is not None and z_tcd < -_ABNORMAL_Z_HIGH:
        combined.append(_pch_combined_card())

    return fired + combined
