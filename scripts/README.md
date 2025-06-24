given timestamp, get the nearest gt pose from gt trajectory in tum.



1. use evo to get the transform matrix between your estimate trajectory and gt trajectory

```bash
evo_ape tum Parkinglot-2023-10-28-18-59-01_0.005_ins_tum.txt optimized_poses_tum.txt -va
```

![image-20250624191617018](./README/image-20250624191617018.png)

2. set the target pose timestamps and the transform matrix

![image-20250624191710157](./README/image-20250624191710157.png)

3. then we can get the corresponding gt pose for the estimated pose.

![image-20250624191759392](./README/image-20250624191759392.png)

![image-20250624191841231](./README/image-20250624191841231.png)

