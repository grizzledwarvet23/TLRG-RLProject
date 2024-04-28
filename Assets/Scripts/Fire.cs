using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fire : MonoBehaviour
{
    Rigidbody2D rb;
    public float speed;

    GameObject player;
    // Start is called before the first frame update
    void Start()
    {
        if(GameObject.Find("Player") != null) {
            player = GameObject.Find("Player");
        }
        rb = GetComponent<Rigidbody2D>();
        StartCoroutine(Die(3, true));
    }

    void FixedUpdate() {
        rb.velocity = transform.right * speed;
    }

    IEnumerator Die(float time, bool missed=false) {
        yield return new WaitForSeconds(time);
        if(missed)
        {
            if(player.GetComponent<PlayerRL>() != null)
            {
                //JUST DO FOR FIRE TRAINING
                player.GetComponent<PlayerRL>().AddRewardExternal(-1f); // to encourage not missing.
            }
        }
        Destroy(gameObject);
    }

    void OnTriggerEnter2D(Collider2D other) {
        if(other.gameObject.tag == "Earth") {
            //increment hitByFireCount
            other.gameObject.GetComponent<Earth>().hitByFireCount++;
            GetComponent<BoxCollider2D>().enabled = false;
            GetComponent<SpriteRenderer>().enabled = false;
            StartCoroutine(Die(1));
        }
        else if (other.gameObject.tag == "Enemy") {
            other.gameObject.GetComponent<Enemy>().TakeDamage(1);
            //given that the enemy is not burning, set them to isBurning with a chance of 30%
            if (!other.gameObject.GetComponent<Enemy>().isBurning && !other.gameObject.GetComponent<Enemy>().isSlowed) {
                if (Random.Range(0, 100) < 30) {
                    other.gameObject.GetComponent<Enemy>().isBurning = true;
                }
            }
            //set collider and sprite off
            GetComponent<BoxCollider2D>().enabled = false;
            GetComponent<SpriteRenderer>().enabled = false;
            StartCoroutine(Die(1));
        }
    }
}
