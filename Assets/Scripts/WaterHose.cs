using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WaterHose : MonoBehaviour
{
    public float pushForce = 10f; // Adjust the force magnitude as needed
    private Rigidbody2D objectRigidbody;

    private void Start()
    {
    }

    private void OnParticleCollision(GameObject other)
    {
        // Check if the collided object has a Collider2D component
        if(other != null) {
            if(other.tag == "Innocent") {
           //other has a circle collider. add a force to that collider based on the direction that the particles are hitting it from:
              Vector3 forceDirection = other.transform.position - transform.position;
                forceDirection.Normalize();
                //add hoirzontal and vertical force based on direction, however, reduce magnitude of vertical force.
                //if velocity x is too high in magnitude, then force in x direction is 0. check rb velocity of other to do this:
                if(Mathf.Abs(other.GetComponent<Rigidbody2D>().velocity.x) > 3) {
                    forceDirection.x = 0;
                }
                //make x direction not be positive ever
                if(forceDirection.x > 0) {
                    forceDirection.x = -forceDirection.x;
                }
                other.GetComponent<Rigidbody2D>().AddForce(new Vector2(forceDirection.x * pushForce * 1.5f, forceDirection.y * pushForce / 2f), ForceMode2D.Impulse);
                
                

            }
            else if(other.tag == "Enemy") {
                int waterCount = other.GetComponent<Enemy>().hitByWaterCount++;
                other.GetComponent<Enemy>().timeLastWaterHit = Time.time;
                if(waterCount > 30) {
                    other.GetComponent<Enemy>().SlowDown();
                    //if is burning, set it to false
                    if(other.GetComponent<Enemy>().isBurning) {
                        other.GetComponent<Enemy>().isBurning = false;
                    }
                }
                //and destroy all particles that hit the enemy. cannot do destroy because this destroys the whole particle system
                
                
            }

            //if tagged earth, increase hitByWaterCount
            else if(other.tag == "Earth" && other.GetComponent<Earth>().isBurning) {
                other.GetComponent<Earth>().hitByWaterCount++;
            }


        }
        
    }
}
