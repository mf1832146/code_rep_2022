WORKDIR="/home/group/tz/method_name_prediction_2022"
export PYTHONPATH=$WORKDIR

TASK=${1}
SUB_TASK=${2}
MODEL_TAG=${3}
GPU=${4}
DATA_NUM=${5}
BS=${6}
LR=${7}
MAX_AST_LEN=${8}
MAX_CODE_LEN=${9}
MAX_DFG_LEN=${10}
MAX_REL_POS=${11}
USE_AST=${12}
USE_CODE=${13}
USE_DFG=${14}
TRG_LEN=${15}
PATIENCE=${16}
EPOCH=${17}
WARMUP=${18}
MODEL_DIR=${19}
SUMMARY_DIR=${20}
RES_FN=${21}

if [[ $DATA_NUM == -1 ]]; then
  DATA_TAG='all'
else
  DATA_TAG=$DATA_NUM
  EPOCH=1
fi

INPUT_TYPE=""
INPUT_CMD=""
if [[ ${USE_AST} == 1 ]]; then
  INPUT_TYPE="${INPUT_TYPE}_use_ast"
  INPUT_CMD="--use_ast  "
fi

if [[ ${USE_CODE} == 1 ]]; then
  INPUT_TYPE="${INPUT_TYPE}_use_code"
  INPUT_CMD="${INPUT_CMD} --use_code "
fi
if [[ ${USE_DFG} == 1 ]]; then
  INPUT_TYPE="${INPUT_TYPE}_use_dfg"
  INPUT_CMD="${INPUT_CMD} --use_dfg "
fi

FULL_MODEL_TAG=${MODEL_TAG}_${INPUT_TYPE}_${DATA_TAG}_lr${LR}_bs${BS}_pat${PATIENCE}_e${EPOCH}

if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
else
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
fi

CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

#if [[ $MODEL_TAG == deberta ]]; then
#  MODEL_TYPE=roberta
#  TOKENIZER=Salesforce/codet5-small
#  MODEL_PATH=roberta-base
#elif [[ $MODEL_TAG == codebert ]]; then
#  MODEL_TYPE=roberta
#  TOKENIZER=roberta-base
#  MODEL_PATH=microsoft/codebert-base
#elif [[ $MODEL_TAG == bart_base ]]; then
#  MODEL_TYPE=bart
#  TOKENIZER=facebook/bart-base
#  MODEL_PATH=facebook/bart-base
#elif [[ $MODEL_TAG == codet5_small ]]; then
#  MODEL_TYPE=codet5
#  TOKENIZER=vocab/codet5-small
#  MODEL_PATH=Salesforce/codet5-small
#elif [[ $MODEL_TAG == codet5_base ]]; then
#  MODEL_TYPE=codet5
#  TOKENIZER=Salesforce/codet5-base
#  MODEL_PATH=Salesforce/codet5-base
#fi
MODEL_TYPE='deberta'

#if [[ ${TASK} == 'clone' ]]; then
#  RUN_FN=${WORKDIR}/run_clone.py
#elif [[ ${TASK} == 'defect' ]] && [[ ${MODEL_TYPE} == 'roberta' ||  ${MODEL_TYPE} == 'bart' ]]; then
#  RUN_FN=${WORKDIR}/run_defect.py
#else
#  RUN_FN=${WORKDIR}/run_gen.py
#fi

RUN_FN=${WORKDIR}/run.py

CUDA_VISIBLE_DEVICES=${GPU} \
  python ${RUN_FN}  \
  --do_train --do_eval --do_eval_acc --do_test  --save_last_checkpoints --always_save_model \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
  --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --cache_path ${CACHE_DIR} --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  --train_batch_size ${BS} --eval_batch_size ${BS} \
  --max_ast_len ${MAX_AST_LEN} --max_code_len ${MAX_CODE_LEN} --max_dfg_len ${MAX_DFG_LEN} --max_rel_pos ${MAX_REL_POS} ${INPUT_CMD} \
  --max_target_length ${TRG_LEN} \
  2>&1 | tee ${LOG}
