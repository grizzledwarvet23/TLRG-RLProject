using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Grain : MonoBehaviour
{
    public float velocity;
    
    private GameObject player;
    

    public enum type {Rice, Bean, Fruit}

    public type grainType;
    // Start is called before the first frame update
    void Start()
    {
        player = GameObject.Find("Player");
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
                    player.GetComponent<PlayerRL>().AddRewardExternal(1);
                }

                GameManager.numCorrect++;
                GameManager.UpdateScoreText();
                GameManager.PlayCorrectSound();
                Destroy(gameObject);

            }
            else {
                if(player.GetComponent<PlayerRL>() != null)
                {
                    player.GetComponent<PlayerRL>().AddRewardExternal(-1);
                }

                GameManager.numIncorrect++;
                GameManager.PlayWrongSound();
                Destroy(gameObject);
            }
        }
    }
}
