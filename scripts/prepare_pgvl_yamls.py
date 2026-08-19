"""Provenance record: how benchmarks/tcga_{brca,nsclc,rcc}/protocol.yaml were
derived from the original multi-cohort benchmarks/tcga/protocol.yaml.

This script no longer runs as-is. benchmarks/tcga/ was deleted once the three
per-cohort protocols superseded it, so the input this reads is gone, and the
paths below are hardcoded to one machine. It is kept because it is the only
record of which feature path each cohort was repointed at and why -- notably
that BRCA's 5x CONCH features live under 5x_512px_0px_overlap rather than the
5x_256px_0px_overlap the original protocol assumed.

Edit the per-cohort protocol.yaml files directly instead.
"""
import yaml
from copy import deepcopy
import os

def prepare():
    os.makedirs('benchmarks/tcga_brca', exist_ok=True)
    os.makedirs('benchmarks/tcga_nsclc', exist_ok=True)
    
    with open('benchmarks/tcga/protocol.yaml', 'r') as f:
        proto = yaml.safe_load(f)
        
    # ------------- BRCA -------------
    proto_brca = deepcopy(proto)
    proto_brca['name'] = 'tcga_brca_resolution_registry'
    if 'nsclc' in proto_brca['cohorts']: del proto_brca['cohorts']['nsclc']
    
    # Patch paths
    fs = proto_brca['feature_sources']
    fs['conch_v1_5x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/5x_512px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['conch_v1_10x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/10x_256px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['conch_v1_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/20x_512px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['keep_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/20x_256px_0px_overlap/features_keep/{slide_id}.h5'
    fs['titan_slide_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/20x_512px_0px_overlap/slide_features_titan/{slide_id}.h5'
    
    with open('benchmarks/tcga_brca/protocol.yaml', 'w') as f:
        proto_str = yaml.dump(proto_brca, sort_keys=False)
        proto_str = proto_str.replace('${PGVL_USER_ROOT}/.cache/huggingface', '${PGVL_STORAGE_ROOT}/.cache_huggingface')
        f.write(proto_str)
        
    # ------------- NSCLC -------------
    proto_nsclc = deepcopy(proto)
    proto_nsclc['name'] = 'tcga_nsclc_resolution_registry'
    if 'brca' in proto_nsclc['cohorts']: del proto_nsclc['cohorts']['brca']
    if 'rcc' in proto_nsclc['cohorts']: del proto_nsclc['cohorts']['rcc']
    
    # Patch paths
    fs = proto_nsclc['feature_sources']
    fs['conch_v1_10x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/10x_256px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['conch_v1_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/20x_512px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['keep_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/20x_256px_0px_overlap/features_keep/{slide_id}.h5'
    fs['titan_slide_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/20x_512px_0px_overlap/slide_features_titan/{slide_id}.h5'
    
    # Remove 5x experiments
    exps = proto_nsclc['experiments']
    if 'focus_5x20x' in exps: del exps['focus_5x20x']
    if 'mscpt_5x20x' in exps: del exps['mscpt_5x20x']
    
    with open('benchmarks/tcga_nsclc/protocol.yaml', 'w') as f:
        proto_str = yaml.dump(proto_nsclc, sort_keys=False)
        proto_str = proto_str.replace('${PGVL_USER_ROOT}/.cache/huggingface', '${PGVL_STORAGE_ROOT}/.cache_huggingface')
        f.write(proto_str)

    # ------------- RCC -------------
    proto_rcc = deepcopy(proto)
    proto_rcc['name'] = 'tcga_rcc_resolution_registry'
    if 'brca' in proto_rcc['cohorts']: del proto_rcc['cohorts']['brca']
    if 'nsclc' in proto_rcc['cohorts']: del proto_rcc['cohorts']['nsclc']
    
    # Patch paths
    fs = proto_rcc['feature_sources']
    fs['conch_v1_10x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/10x_256px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['conch_v1_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/20x_512px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['keep_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/20x_256px_0px_overlap/features_keep/{slide_id}.h5'
    fs['titan_slide_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/trident_features/master_benchmark/20x_512px_0px_overlap/slide_features_titan/{slide_id}.h5'
    
    if 'rcc' in proto_rcc['cohorts']:
        proto_rcc['cohorts']['rcc']['metadata_availability'] = 'future'

    os.makedirs('benchmarks/tcga_rcc', exist_ok=True)
    with open('benchmarks/tcga_rcc/protocol.yaml', 'w') as f:
        proto_str = yaml.dump(proto_rcc, sort_keys=False)
        proto_str = proto_str.replace('${PGVL_USER_ROOT}/.cache/huggingface', '${PGVL_STORAGE_ROOT}/.cache_huggingface')
        f.write(proto_str)

    # ------------- ADDITIONAL TASKS -------------
    with open('benchmarks/additional_tasks/protocol.yaml', 'r') as f:
        proto_addl = yaml.safe_load(f)
        
    fs = proto_addl['feature_sources']
    fs['conch_v1_5x']['path_template'] = '${PGVL_STORAGE_ROOT}/UBC-OCEAN/5x_256px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['conch_v1_10x']['path_template'] = '${PGVL_STORAGE_ROOT}/UBC-OCEAN/10x_256px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['conch_v1_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/UBC-OCEAN/20x_256px_0px_overlap/features_conch_v1/{slide_id}.h5'
    fs['keep_20x']['path_template'] = '${PGVL_STORAGE_ROOT}/UBC-OCEAN/20x_256px_0px_overlap/features_keep/{slide_id}.h5'
    
    if 'titan_slide_20x' in fs: del fs['titan_slide_20x']
    if 'sldpc' in proto_addl['experiments']: del proto_addl['experiments']['sldpc']
    
    if 'camelyon16' in proto_addl['cohorts']:
        proto_addl['cohorts']['camelyon16']['metadata_availability'] = 'future'
    
    with open('benchmarks/additional_tasks/protocol.yaml', 'w') as f:
        proto_str = yaml.dump(proto_addl, sort_keys=False)
        proto_str = proto_str.replace('${PGVL_USER_ROOT}/.cache/huggingface', '${PGVL_STORAGE_ROOT}/.cache_huggingface')
        f.write(proto_str)

if __name__ == "__main__":
    prepare()
    print("YAMLs prepped!")
