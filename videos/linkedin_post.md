# LinkedIn post (video: `linkedin_cut_en.mp4`, 65 s)

## English

Teaching a robot arm to pick and place, with a GPU motion planner as the teacher.

Inspired by what I saw at ROSCon China, on a single RTX 4070 Ti I built an SO-101 pick-and-place pipeline in Isaac Sim:

- NVIDIA cuRobo (the GPU motion-planning library behind Isaac ROS cuMotion) plans grasps and transports in about 0.1 s, and replans in 0.14 s when an obstacle appears mid-transport.
- I used cuRobo as an expert to generate 120 physics-verified demonstrations in simulation.
- I fine-tuned a SmolVLA policy on those demos plus 45 public real-robot demos. On the same 20 test scenes, success went from 0/20 (real data only) to 7/20 (with sim demos). The cuRobo expert scored 20/20.
- Combining both, with cuRobo for approach and transport and the learned policy for the grasp, reached 16/20, including replanning around obstacles.

All results are in simulation. It has not been tested on real SO-101 hardware yet.

Code: https://github.com/sionchu/curobo-so101-demo
Stack: NVIDIA cuRobo, Isaac Sim, LeRobot (SmolVLA). Real demo data: lerobot/svla_so101_pickplace (Apache-2.0).

#ROSCon #robotics #NVIDIA #cuRobo #IsaacSim #LeRobot #imitationlearning

## 한국어

GPU 모션 플래너를 선생님 삼아 로봇팔에게 집어서 옮기기를 가르쳐 봤습니다.

ROSCon China에서 본 내용에서 영감을 받아, RTX 4070 Ti 한 대로 Isaac Sim에서 SO-101 로봇팔의 집어 옮기기 파이프라인을 만들었습니다.

- NVIDIA cuRobo(Isaac ROS cuMotion의 바탕이 되는 GPU 모션 계획 라이브러리)가 집기와 운반 경로를 약 0.1초 만에 계획합니다. 운반 중에 장애물이 나타나면 0.14초 만에 다시 계획합니다.
- cuRobo를 전문가로 삼아 시뮬레이션에서 물리 검증을 통과한 시연 120개를 만들었습니다.
- 이 시연과 공개된 실제 로봇 시연 45개로 SmolVLA 정책을 학습했습니다. 같은 테스트 장면 20개에서 실제 데이터만 쓰면 0/20이었는데, 시뮬레이션 시연을 더하자 7/20이 됐습니다. 전문가인 cuRobo는 20/20입니다.
- 둘을 합쳐 cuRobo가 접근과 운반을, 학습한 정책이 집기를 맡게 하자 장애물 재계획을 포함해 16/20을 달성했습니다.

모든 결과는 시뮬레이션에서 나온 것이고, 실제 SO-101 하드웨어에서는 아직 시험하지 않았습니다.

코드: https://github.com/sionchu/curobo-so101-demo
사용 도구: NVIDIA cuRobo, Isaac Sim, LeRobot(SmolVLA). 실제 시연 데이터: lerobot/svla_so101_pickplace (Apache-2.0)

#ROSCon #로보틱스 #robotics #NVIDIA #cuRobo #IsaacSim #LeRobot #모방학습
