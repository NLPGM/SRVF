
k_shots=(5)
#k_shots=(5 10 20 50)

retrieval_stage1_modes=('task_specific_retrieval')
#retrieval_stage1_modes=('random' 'simcse_retrieval' 'task_specific_retrieval')

for k_shot in "${k_shots[@]}"
do

  CUDA_VISIBLE_DEVICES=3 python rational_initialization.py  --dataset SemEval \
                                                    --k_shot $k_shot \
                                                    --llm_type llama2_7b_chat

  CUDA_VISIBLE_DEVICES=3 python do_intervention.py  --dataset SemEval \
                                                    --k_shot $k_shot \
                                                    --llm_type llama2_7b_chat

  CUDA_VISIBLE_DEVICES=3 python train_task_specific_retriever.py  --dataset SemEval  --k_shot $k_shot

  CUDA_VISIBLE_DEVICES=3 python train_rational_supervisor.py  --dataset SemEval \
                                                              --k_shot $k_shot \
                                                              --llm_type llama2_7b_chat \
                                                              --train_epochs 50

  # srvf 方法
  for retrieval_stage1_mode in "${retrieval_stage1_modes[@]}"
  do
      CUDA_VISIBLE_DEVICES=3 python main.py --dataset SemEval \
                                            --k_shot $k_shot \
                                            --retrieval_stage1_mode $retrieval_stage1_mode \
                                            --feedback_topk 5 \
                                            --llm_type llama2_7b_chat \
                                            --do_stage1

      CUDA_VISIBLE_DEVICES=3 python main.py --dataset SemEval \
                                            --k_shot $k_shot \
                                            --retrieval_stage1_mode $retrieval_stage1_mode \
                                            --feedback_topk 5 \
                                            --llm_type llama2_7b_chat \
                                            --do_stage2
  done

done