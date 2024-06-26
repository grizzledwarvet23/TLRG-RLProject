using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Grain : MonoBehaviour
{
    public float velocity;

    private float test_velocity;
    
    private GameObject player;

    private int k = 0;
    private int c = 0;

    public enum type {Rice, Bean, Fruit}

    public type grainType;
    // Start is called before the first frame update
    void Start()
    {
        player = GameObject.Find("Player");
        // test_velocity = velocity * 8;
    }

    void FixedUpdate() {
        transform.position += Vector3.down * velocity * Time.deltaTime;
    }

    // void OnCollisionEnter(Collision collision) {
    //     if (collision.gameObject.tag == "Bin") {
    //         //check that this type matches the bin name: as in the bin name will start with "Rice" or "Bean" or "Fruit"
    //         if (collision.gameObject.name.StartsWith(grainType.ToString())) {
    //             //add to the bin
    //             //collision.gameObject.GetComponent<Bin>().AddGrain(this);
    //             GameManager.numCorrect++;

    //         }
    //         else {
    //             //destroy the grain
    //             GameManager.numIncorrect++;
    //             Destroy(gameObject);
    //         }
    //     }
    // }
    //do the above for OnTriggerEnter2D
    void OnTriggerEnter2D(Collider2D collision) {
        if (collision.gameObject.tag == "Bin") {
            //check that this type matches the bin name: as in the bin name will start with "Rice" or "Bean" or "Fruit"
            if (collision.gameObject.name.StartsWith(grainType.ToString())) {
                //add to the bin
                //collision.gameObject.GetComponent<Bin>().AddGrain(this);

                if(player.GetComponent<PlayerRL>() != null)
                {
                    player.GetComponent<PlayerRL>().AddRewardExternal(15 + k);
                    k += 2;
                }

                GameManager.numCorrect++;
                GameManager.UpdateScoreText();
                GameManager.PlayCorrectSound();
                Destroy(gameObject);

            }
            else {
                if(player.GetComponent<PlayerRL>() != null)
                {
                    player.GetComponent<PlayerRL>().AddRewardExternal(-10 + c);
                    c-=5;
                }

                GameManager.numIncorrect++;
                GameManager.PlayWrongSound();
                Destroy(gameObject);
            }
        }
    }
}
