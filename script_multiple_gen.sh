#!/usr/bin/env bash

CKPT_DIR="/xuqianxun/Wan2.1-T2V-1.3B"
SIZE="480*832"
SEED=42
SAMPLER="unipc"
GUIDE=3.5
TAR_GUIDE=10.5
SHIFT=12
STEPS=50
N_MAX=40
N_MIN=0
N_AVG=5
WORSE_AVG=2
OMEGA=3
WINDOW=11
DECAY=0.1
FRAMES=41

run_edit () {
  local VIDEO=$1
  local FULL_PROMPT=$2
  local SRC_WORD=$3
  local TGT_WORD=$4

  local SRC_PROMPT="${FULL_PROMPT//${TGT_WORD}/${SRC_WORD}}"
  local TGT_PROMPT="${FULL_PROMPT}"

  echo -e "\n▶ Editing ${VIDEO}  –  ${SRC_WORD} ➜ ${TGT_WORD}"

  python edit.py \
    --task t2v-1.3B \
    --size ${SIZE} \
    --base_seed ${SEED} \
    --ckpt_dir ${CKPT_DIR} \
    --sample_solver ${SAMPLER} \
    --source_video_path "video_list/${VIDEO}" \
    --source_prompt "${SRC_PROMPT}" \
    --source_words "${SRC_WORD}" \
    --prompt "${TGT_PROMPT}" \
    --target_words "${TGT_WORD}" \
    --sample_guide_scale ${GUIDE} \
    --tar_guide_scale   ${TAR_GUIDE} \
    --sample_shift      ${SHIFT} \
    --sample_steps      ${STEPS} \
    --n_max     ${N_MAX} \
    --n_min     ${N_MIN} \
    --n_avg     ${N_AVG} \
    --worse_avg ${WORSE_AVG} \
    --omega      ${OMEGA} \
    --window_size   ${WINDOW} \
    --decay_factor ${DECAY} \
    --frame_num    ${FRAMES}
}

# Each line: video.mp4, full edited prompt, source_word, target_word
#run_edit "puppy.mp4"        "A car wearing a witch hat in a forest of golden autumn leaves. The camera stays fixed." "small puppy" "car"
#run_edit "cat_box.mp4"      "Two cars playing boxing in a boxing field. The camera stays fixed." "cats" "cars"
#run_edit "bear_g.mp4"       "A woman is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery. The camera remains fixed." "brown bear" "woman"
#run_edit "173.mp4" "A teddy bear dancing in a studio. The camera stays fixed." "girl" "teddy bear"
#run_edit "dog_flower_g.mp4" "A pixar telephone sniffing flowers in a green garden. The camera stays fixed." "golden retriever" "pixar telephone"
#run_edit "sea_turtle.mp4"   "A woman swimming gracefully under clear blue water. The camera stays fixed." "sea turtle" "woman"
#run_edit "jeep.mp4"         "A ship driving on a road with green trees on the side. The camera stays fixed." "rugged jeep" "ship"
#run_edit "rabbit.mp4"       "A little bug eating a slice of watermelon. The camera stays fixed." "black rabbit" "little bug"
#run_edit "wolf.mp4"         "A boy trotting through a green forest. The camera stays fixed." "grey wolf" "boy"
#run_edit "cockatiel.mp4"    "A small quad-rotor drone flapping its wings on a tree branch. The camera stays fixed." "red parrot" "small quad-rotor drone"
#run_edit "sea_lion.mp4"     "A small yellow submarine basking on a rock at the seaside. The camera stays fixed." "sea lion" "small yellow submarine"
#run_edit "girl_and_dog.mp4" "A young girl and boy sitting in a forest. The camera stays fixed." "dog" "boy"
#run_edit "woman.mp4"    "A bear walking in a park." "woman" "bear"
run_edit "dino.mp4"    "A parrot running out of a gate and steps into the wild, under the sun." "dinosaur" "parrot"
#run_edit "gym_woman.mp4"        "A chicken running on the treadmill." "woman" "chicken"
#run_edit "gym_woman.mp4"        "A dog running on the treadmill facing the window." "woman" "dog"
#run_edit "gym_woman.mp4"        "A baby running on the treadmill." "woman" "baby"
#run_edit "blackswan.mp4"    "An origami boat gliding across a calm lake. The camera stays fixed." "black swan" "origami boat"

echo -e "\n✅ All edits launched with consistent prompt structure."